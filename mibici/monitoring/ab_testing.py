"""
Módulo de entrenamiento de versión alternativa (v3), registro en MLflow Model Registry
y evaluación metodológica mediante A/B Testing (Variante A: v2 vs. Variante B: v3)
sobre el periodo actual de producción (Octubre 2026) para P1, P2 y P3 (Equipo 35).

Metodología estadística (remuestreo bootstrap por bloques):
  - En lugar de asumir que las 208,912 observaciones estación x hora son independientes
    (lo cual subestima el error estándar y exagera la significancia estadística debido a
    la autocorrelación horaria y a efectos comunes de red por día o por estación), la
    incertidumbre se estima mediante bootstrap por bloques:
      * P1 y P2: remuestreo por bloques de días (31 días de octubre 2026) y por bloques
        de estaciones (469 estaciones activas).
      * P3: remuestreo por bloques de estaciones (469 estaciones activas).
  - En cada caso se reporta la diferencia (v3 - v2), el error estándar bootstrap, el
    intervalo de confianza al 95% (percentiles 2.5 y 97.5) y el p-valor bilateral de
    remuestreo con corrección de continuidad.

Cómo ejecutarlo:
    python -m mibici.monitoring.ab_testing
"""

import json
from pathlib import Path
import time

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mibici import config, features
from mibici.modeling import train
from mibici.monitoring import drift_report

matplotlib.use("Agg")

# Número de réplicas bootstrap por bloques (semilla fija tomada de config.RANDOM_SEED)
N_BOOTSTRAP_BLOQUES = 2000
N_BOOTSTRAP_PR_AUC = 300
N_BOOTSTRAP_SILHOUETTE = 500


# ==============================================================================
# 1. PREPARAR DATOS DE REENTRENAMIENTO (OCT 2022 - SEP 2026 CON NOMENCLATURA 10)
# ==============================================================================
def preparar_datos_reentrenamiento_v3() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepara los conjuntos de entrenamiento actualizados para la versión v3 (Variante B):
      - Para P1 y P2: Toma los 48 meses previos a Octubre 2026 (Oct 2022 - Sep 2026),
        incorporando los 9 meses de 2026 (Ene - Sep 2026) que v2 no vio al entrenarse,
        y actualiza las capacidades de estaciones según nomenclatura_2026_10.csv.
      - Para P3: Calcula los perfiles de estaciones sobre el bloque reciente de 2026
        previo a octubre (Enero - Septiembre 2026) con nomenclatura_2026_10.csv para
        capturar las 469 estaciones activas y los hábitos actuales de viaje en el AMG.
    """
    df_nom_10 = features.cargar_nomenclatura(
        ruta_nomenclatura=config.NOMENCLATURE_FILE_2026_10
    )
    mapa_capacidad_10 = dict(
        zip(
            df_nom_10["Estacion_Id"].astype("int64"),
            df_nom_10["dpcapacity"].astype("float64"),
        )
    )
    estaciones_vigentes_10 = set(df_nom_10["Estacion_Id"].astype("int64").tolist())

    # 1. Datos históricos P1 y P2 (Octubre 2022 - Septiembre 2026)
    df_p1_p2_hist = pd.read_parquet(config.INTERIM_P1_P2_FILE)
    df_p1_p2_v3 = df_p1_p2_hist.loc[
        df_p1_p2_hist["Estacion_Id"].isin(estaciones_vigentes_10)
    ].copy()

    # Actualizamos dpcapacity, lag_1_ratio_capacidad y en_riesgo con el catálogo 2026_10
    df_p1_p2_v3["dpcapacity"] = (
        df_p1_p2_v3["Estacion_Id"].map(mapa_capacidad_10).astype("float64")
    )
    cap_segura = df_p1_p2_v3["dpcapacity"].replace(0, np.nan)
    df_p1_p2_v3["lag_1_ratio_capacidad"] = (
        df_p1_p2_v3["lag_1_flujo_neto"].abs() / cap_segura
    )
    ratio_actual = df_p1_p2_v3["flujo_neto"].abs() / cap_segura
    df_p1_p2_v3["en_riesgo"] = (
        ratio_actual >= config.RISK_CAPACITY_RATIO_THRESHOLD
    ).astype("int8")
    df_p1_p2_v3 = df_p1_p2_v3.dropna(
        subset=drift_report.COLUMNAS_FEATURES_P1_P2
    ).reset_index(drop=True)

    # 2. Datos de perfiles de estaciones P3 (Enero - Septiembre 2026 con nomenclatura 2026_10)
    columnas_viajes = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    df_viajes = pd.read_parquet(config.PARQUET_FILE, columns=columnas_viajes)
    viajes_2026 = df_viajes.loc[
        df_viajes["Inicio_del_viaje"] >= pd.Timestamp("2026-01-01 00:00:00")
    ]
    df_p3_v3 = features.generar_features_p3_clustering(
        viajes_2026,
        ruta_nomenclatura=config.NOMENCLATURE_FILE_2026_10,
    )

    return df_p1_p2_v3, df_p3_v3


# ==============================================================================
# 2. REGISTRAR O RECUPERAR LA VERSIÓN v3 EN EL MODEL REGISTRY DE MLFLOW
# ==============================================================================
def obtener_o_entrenar_version_v3() -> dict:
    """
    Verifica si la versión 3 (Variante B reentrenada) ya está registrada en el
    Model Registry de MLflow para P1, P2 y P3. Si aún no existe, entrena los
    modelos v3, guarda sus corridas en MLflow y los registra como versión 3 en
    el Model Registry.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    cliente = MlflowClient()
    for run_f in cliente.search_runs(
        [experimento.experiment_id], "attributes.status = 'FAILED'"
    ):
        cliente.delete_run(run_f.info.run_id)

    git_commit, data_version_base = train.obtener_metadatos_trazabilidad()
    _, _, manifiesto_10 = drift_report.construir_features_octubre_2026()
    data_version_combinada = f"{data_version_base}+{manifiesto_10['data_version_short']}"

    # Revisamos si los 3 modelos ya cuentan con la versión 3 registrada
    versiones_p1 = [
        int(mv.version)
        for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P1}'")
    ]
    versiones_p2 = [
        int(mv.version)
        for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P2}'")
    ]
    versiones_p3 = [
        int(mv.version)
        for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P3}'")
    ]

    if 3 in versiones_p1 and 3 in versiones_p2 and 3 in versiones_p3:
        print("La versión v3 (Variante B) ya se encuentra registrada en el Model Registry.")
        return {
            "p1_v3_version": "3",
            "p2_v3_version": "3",
            "p3_v3_version": "3",
            "data_version_v3": data_version_combinada,
        }

    print("\nEntrenando versión alternativa v3 (Oct 2022 - Sep 2026 + nomenclatura_2026_10)...")
    df_p1_p2_v3, df_p3_v3 = preparar_datos_reentrenamiento_v3()
    cols_x_p1_p2 = drift_report.COLUMNAS_FEATURES_P1_P2
    cols_x_p3 = drift_report.COLUMNAS_FEATURES_P3

    X_train_p1_p2 = df_p1_p2_v3[cols_x_p1_p2].astype("float64")
    y_train_p1 = df_p1_p2_v3["viajes_iniciados"].astype("float64")
    y_train_p2 = df_p1_p2_v3["en_riesgo"].astype("int64")

    # --------------------------------------------------------------------------
    # A) Entrenar y registrar v3 para P1 (Regresión de demanda horaria)
    # --------------------------------------------------------------------------
    if 3 not in versiones_p1:
        t0 = time.time()
        modelo_p1_v3 = HistGradientBoostingRegressor(
            max_iter=300,
            max_depth=12,
            learning_rate=0.06,
            min_samples_leaf=30,
            random_state=config.RANDOM_SEED,
        )
        pesos_p1 = np.where(
            df_p1_p2_v3["Fecha_Hora"] >= pd.Timestamp("2026-01-01 00:00:00"),
            3.0,
            1.0,
        )
        modelo_p1_v3.fit(X_train_p1_p2, y_train_p1, sample_weight=pesos_p1)
        pred_muestra_p1 = np.clip(modelo_p1_v3.predict(X_train_p1_p2.head(200)), 0, None)

        nombre_run_p1 = "P1_Retrain_v3_HistGB_iter300_d12_lr0.06_Oct2022_Sep2026"
        with mlflow.start_run(run_name=nombre_run_p1) as run_p1:
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "git_commit": git_commit,
                    "data_version": data_version_combinada,
                    "question": "p1",
                    "model_family": "GradientBoosting",
                    "stage_of_work": "continuous_training_v3",
                    "ab_variant": "B",
                }
            )
            mlflow.log_params(
                {
                    "question": "p1",
                    "model_name": "hgb_retrained_v3",
                    "train_window": "2022-10_a_2026-09 (48 meses + pesos 2026)",
                    "nomenclature": "nomenclatura_2026_10.csv",
                    "n_estimators": 300,
                    "max_depth": 12,
                    "learning_rate": 0.06,
                    "recent_2026_sample_weight": 3.0,
                    "random_seed": config.RANDOM_SEED,
                }
            )
            mlflow.log_metric("training_time_sec", round(time.time() - t0, 2))
            firma_p1 = infer_signature(X_train_p1_p2.head(50), pred_muestra_p1[:50])
            info_m1 = mlflow.sklearn.log_model(
                sk_model=modelo_p1_v3,
                name="model",
                signature=firma_p1,
                input_example=X_train_p1_p2.head(5),
                serialization_format="cloudpickle",
            )
            ver_p1 = mlflow.register_model(
                model_uri=info_m1.model_uri, name=config.MODEL_NAME_P1
            )
            cliente.update_model_version(
                name=config.MODEL_NAME_P1,
                version=ver_p1.version,
                description=(
                    f"Versión v{ver_p1.version} (Variante B - Continuous Training). "
                    "Reentrenada con 48 meses (Oct 2022 - Sep 2026) y nomenclatura_2026_10.csv. "
                    f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} "
                    f"| Run ID: {run_p1.info.run_id} | Git Commit: {git_commit} "
                    f"| Data Version: {data_version_combinada}."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P1, ver_p1.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P1,
                ver_p1.version,
                "data_version",
                data_version_combinada,
            )
            print(
                f"  -> Registrado {config.MODEL_NAME_P1} v{ver_p1.version} (run_id={run_p1.info.run_id})"
            )

    # --------------------------------------------------------------------------
    # B) Entrenar y registrar v3 para P2 (Clasificación de riesgo de desbalance)
    # --------------------------------------------------------------------------
    if 3 not in versiones_p2:
        t0 = time.time()
        modelo_p2_v3 = HistGradientBoostingClassifier(
            class_weight="balanced",
            max_iter=300,
            max_depth=12,
            learning_rate=0.06,
            min_samples_leaf=30,
            random_state=config.RANDOM_SEED,
        )
        pesos_p2 = np.where(
            df_p1_p2_v3["Fecha_Hora"] >= pd.Timestamp("2026-01-01 00:00:00"),
            2.0,
            1.0,
        )
        modelo_p2_v3.fit(X_train_p1_p2, y_train_p2, sample_weight=pesos_p2)
        pred_muestra_p2 = modelo_p2_v3.predict(X_train_p1_p2.head(200))

        nombre_run_p2 = "P2_Retrain_v3_HistGB_Clf_iter300_d12_lr0.06_Oct2022_Sep2026"
        with mlflow.start_run(run_name=nombre_run_p2) as run_p2:
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "git_commit": git_commit,
                    "data_version": data_version_combinada,
                    "question": "p2",
                    "model_family": "GradientBoosting",
                    "stage_of_work": "continuous_training_v3",
                    "ab_variant": "B",
                }
            )
            mlflow.log_params(
                {
                    "question": "p2",
                    "model_name": "hgb_clf_retrained_v3",
                    "train_window": "2022-10_a_2026-09 (48 meses + pesos 2026)",
                    "nomenclature": "nomenclatura_2026_10.csv",
                    "n_estimators": 300,
                    "max_depth": 12,
                    "learning_rate": 0.06,
                    "class_weight": "balanced",
                    "random_seed": config.RANDOM_SEED,
                }
            )
            mlflow.log_metric("training_time_sec", round(time.time() - t0, 2))
            firma_p2 = infer_signature(X_train_p1_p2.head(50), pred_muestra_p2[:50])
            info_m2 = mlflow.sklearn.log_model(
                sk_model=modelo_p2_v3,
                name="model",
                signature=firma_p2,
                input_example=X_train_p1_p2.head(5),
                serialization_format="cloudpickle",
            )
            ver_p2 = mlflow.register_model(
                model_uri=info_m2.model_uri, name=config.MODEL_NAME_P2
            )
            cliente.update_model_version(
                name=config.MODEL_NAME_P2,
                version=ver_p2.version,
                description=(
                    f"Versión v{ver_p2.version} (Variante B - Continuous Training). "
                    "Reentrenada con 48 meses (Oct 2022 - Sep 2026) y nomenclatura_2026_10.csv. "
                    f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} "
                    f"| Run ID: {run_p2.info.run_id} | Git Commit: {git_commit} "
                    f"| Data Version: {data_version_combinada}."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P2, ver_p2.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P2,
                ver_p2.version,
                "data_version",
                data_version_combinada,
            )
            print(
                f"  -> Registrado {config.MODEL_NAME_P2} v{ver_p2.version} (run_id={run_p2.info.run_id})"
            )

    # --------------------------------------------------------------------------
    # C) Entrenar y registrar v3 para P3 (Clustering de estaciones)
    # --------------------------------------------------------------------------
    if 3 not in versiones_p3:
        t0 = time.time()
        X_train_p3 = df_p3_v3[cols_x_p3].astype("float64")
        modelo_p3_v3 = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clusterer",
                    KMeans(
                        n_clusters=4,
                        n_init=20,
                        random_state=config.RANDOM_SEED,
                    ),
                ),
            ]
        )
        modelo_p3_v3.fit(X_train_p3)
        pred_muestra_p3 = modelo_p3_v3.predict(X_train_p3)

        nombre_run_p3 = "P3_Retrain_v3_KMeans_K4_Perfiles2026_Nom10"
        with mlflow.start_run(run_name=nombre_run_p3) as run_p3:
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "git_commit": git_commit,
                    "data_version": data_version_combinada,
                    "question": "p3",
                    "model_family": "KMeans",
                    "stage_of_work": "continuous_training_v3",
                    "ab_variant": "B",
                }
            )
            mlflow.log_params(
                {
                    "question": "p3",
                    "model_name": "kmeans_retrained_v3",
                    "train_window": "2026-01_a_2026-09 (469 estaciones activas)",
                    "nomenclature": "nomenclatura_2026_10.csv",
                    "n_clusters": 4,
                    "n_init": 20,
                    "random_seed": config.RANDOM_SEED,
                }
            )
            mlflow.log_metric("training_time_sec", round(time.time() - t0, 2))
            firma_p3 = infer_signature(X_train_p3.head(50), pred_muestra_p3[:50])
            info_m3 = mlflow.sklearn.log_model(
                sk_model=modelo_p3_v3,
                name="model",
                signature=firma_p3,
                input_example=X_train_p3.head(5),
                serialization_format="cloudpickle",
            )
            ver_p3 = mlflow.register_model(
                model_uri=info_m3.model_uri, name=config.MODEL_NAME_P3
            )
            cliente.update_model_version(
                name=config.MODEL_NAME_P3,
                version=ver_p3.version,
                description=(
                    f"Versión v{ver_p3.version} (Variante B - Continuous Training). "
                    "Reentrenada con perfiles 2026 (469 estaciones) y nomenclatura_2026_10.csv. "
                    f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} "
                    f"| Run ID: {run_p3.info.run_id} | Git Commit: {git_commit} "
                    f"| Data Version: {data_version_combinada}."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P3, ver_p3.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P3,
                ver_p3.version,
                "data_version",
                data_version_combinada,
            )
            print(
                f"  -> Registrado {config.MODEL_NAME_P3} v{ver_p3.version} (run_id={run_p3.info.run_id})"
            )

    return {
        "p1_v3_version": "3",
        "p2_v3_version": "3",
        "p3_v3_version": "3",
        "data_version_v3": data_version_combinada,
    }


# ==============================================================================
# 3. CARGAR UNA VERSIÓN ESPECÍFICA (v2 o v3) DESDE EL MODEL REGISTRY DE MLFLOW
# ==============================================================================
def cargar_version_modelo_registry(nombre_modelo: str, version: str):
    """
    Recupera el estimador subyacente de Scikit-Learn directamente desde el Model
    Registry de MLflow usando su URI oficial 'models:/<nombre_modelo>/<version>'.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    uri_version = f"models:/{nombre_modelo}/{version}"
    return mlflow.sklearn.load_model(uri_version)


# ==============================================================================
# 4. FUNCIONES DE REMUESTREO BOOTSTRAP POR BLOQUES (DÍAS Y ESTACIONES)
# ==============================================================================
def resumir_distribucion_bootstrap(
    diferencias: np.ndarray,
    diff_observada: float,
    n_bloques: int,
    unidad_bloque: str,
    metrica: str,
    direccion: str,
) -> dict:
    """
    Resume la distribución empírica de diferencias (v3 - v2) obtenida mediante
    bootstrap por bloques:
      - Diferencia observada en el conjunto completo y media bootstrap.
      - Error estándar bootstrap (desviación estándar de las réplicas).
      - Intervalo de confianza al 95% por percentiles [2.5%, 97.5%].
      - p-valor bilateral con corrección de continuidad:
        p = min(1, 2 * (min(conteo <= 0, conteo >= 0) + 1) / (B + 1)).
    """
    n_iter = int(len(diferencias))
    media_boot = float(np.mean(diferencias))
    error_estandar = float(np.std(diferencias, ddof=1))
    ic95_inf = float(np.percentile(diferencias, 2.5))
    ic95_sup = float(np.percentile(diferencias, 97.5))

    conteo_menor_igual_cero = int(np.sum(diferencias <= 0.0))
    conteo_mayor_igual_cero = int(np.sum(diferencias >= 0.0))
    p_valor = float(
        min(
            1.0,
            2.0
            * (min(conteo_menor_igual_cero, conteo_mayor_igual_cero) + 1)
            / (n_iter + 1),
        )
    )
    # Es significativo al 95% si el intervalo de confianza no incluye el cero
    significativo_95 = bool((ic95_inf > 0.0) or (ic95_sup < 0.0))

    return {
        "metric": metrica,
        "direction": direccion,
        "block_unit": unidad_bloque,
        "n_blocks": int(n_bloques),
        "n_bootstrap": n_iter,
        "observed_diff": round(float(diff_observada), 6),
        "bootstrap_mean_diff": round(media_boot, 6),
        "bootstrap_std_error": round(error_estandar, 6),
        "ci95_lower": round(ic95_inf, 6),
        "ci95_upper": round(ic95_sup, 6),
        "p_value": round(p_valor, 6),
        "significant_ci95": significativo_95,
    }


def bootstrap_bloques_media_error(
    error_v2: np.ndarray,
    error_v3: np.ndarray,
    bloques: np.ndarray,
    unidad_bloque: str,
    metrica: str,
    n_bootstrap: int = N_BOOTSTRAP_BLOQUES,
    semilla: int = config.RANDOM_SEED,
) -> dict:
    """
    Ejecuta bootstrap por bloques para métricas expresables como promedio de pérdidas
    por fila (MAE en P1, Brier Score en P2 y distancia media al centroide en P3).

    Paso a paso:
      1. Agrupa las observaciones por bloque (día o estación) sumando el error de v2,
         el error de v3 y el número de registros en cada bloque.
      2. En cada réplica bootstrap, selecciona bloques completos con reemplazo.
      3. Recalcula la métrica global exacta de la muestra remuestreada y guarda la
         diferencia (metrica_v3 - metrica_v2).
    """
    df_bloques = (
        pd.DataFrame(
            {
                "bloque": bloques,
                "error_v2": np.asarray(error_v2, dtype="float64"),
                "error_v3": np.asarray(error_v3, dtype="float64"),
            }
        )
        .groupby("bloque", sort=True)
        .agg(
            n_obs=("error_v2", "size"),
            suma_v2=("error_v2", "sum"),
            suma_v3=("error_v3", "sum"),
        )
    )

    n_obs = df_bloques["n_obs"].to_numpy(dtype="float64")
    suma_v2 = df_bloques["suma_v2"].to_numpy(dtype="float64")
    suma_v3 = df_bloques["suma_v3"].to_numpy(dtype="float64")
    n_bloques = len(df_bloques)

    diff_obs = float(np.mean(error_v3) - np.mean(error_v2))
    rng = np.random.default_rng(semilla)
    diferencias = np.empty(n_bootstrap, dtype="float64")

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_bloques, size=n_bloques)
        total_obs = float(n_obs[idx].sum())
        media_v2 = float(suma_v2[idx].sum()) / total_obs
        media_v3 = float(suma_v3[idx].sum()) / total_obs
        diferencias[i] = media_v3 - media_v2

    return resumir_distribucion_bootstrap(
        diferencias=diferencias,
        diff_observada=diff_obs,
        n_bloques=n_bloques,
        unidad_bloque=unidad_bloque,
        metrica=metrica,
        direccion="v3 - v2 (negativo favorece a v3)",
    )


def bootstrap_bloques_f1(
    y_real: np.ndarray,
    pred_v2: np.ndarray,
    pred_v3: np.ndarray,
    bloques: np.ndarray,
    unidad_bloque: str,
    n_bootstrap: int = N_BOOTSTRAP_BLOQUES,
    semilla: int = config.RANDOM_SEED,
) -> dict:
    """
    Ejecuta bootstrap por bloques exacto para la diferencia de F1-Score (v3 - v2) en P2.
    Preagrega TP, FP y FN por bloque (día o estación) para calcular el F1 global exacto
    en cada réplica sin distorsionar el numerador ni el denominador.
    """
    y_arr = np.asarray(y_real, dtype="int64")
    p2_arr = np.asarray(pred_v2, dtype="int64")
    p3_arr = np.asarray(pred_v3, dtype="int64")

    df_bloques = (
        pd.DataFrame(
            {
                "bloque": bloques,
                "tp_v2": ((y_arr == 1) & (p2_arr == 1)).astype("int64"),
                "fp_v2": ((y_arr == 0) & (p2_arr == 1)).astype("int64"),
                "fn_v2": ((y_arr == 1) & (p2_arr == 0)).astype("int64"),
                "tp_v3": ((y_arr == 1) & (p3_arr == 1)).astype("int64"),
                "fp_v3": ((y_arr == 0) & (p3_arr == 1)).astype("int64"),
                "fn_v3": ((y_arr == 1) & (p3_arr == 0)).astype("int64"),
            }
        )
        .groupby("bloque", sort=True)
        .sum()
    )

    tp_v2 = df_bloques["tp_v2"].to_numpy(dtype="float64")
    fp_v2 = df_bloques["fp_v2"].to_numpy(dtype="float64")
    fn_v2 = df_bloques["fn_v2"].to_numpy(dtype="float64")
    tp_v3 = df_bloques["tp_v3"].to_numpy(dtype="float64")
    fp_v3 = df_bloques["fp_v3"].to_numpy(dtype="float64")
    fn_v3 = df_bloques["fn_v3"].to_numpy(dtype="float64")
    n_bloques = len(df_bloques)

    f1_obs_v2 = float(f1_score(y_arr, p2_arr, zero_division=0))
    f1_obs_v3 = float(f1_score(y_arr, p3_arr, zero_division=0))
    diff_obs = f1_obs_v3 - f1_obs_v2

    rng = np.random.default_rng(semilla)
    diferencias = np.empty(n_bootstrap, dtype="float64")

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_bloques, size=n_bloques)

        denom_v2 = float(2.0 * tp_v2[idx].sum() + fp_v2[idx].sum() + fn_v2[idx].sum())
        f1_r_v2 = (2.0 * float(tp_v2[idx].sum()) / denom_v2) if denom_v2 > 0.0 else 0.0

        denom_v3 = float(2.0 * tp_v3[idx].sum() + fp_v3[idx].sum() + fn_v3[idx].sum())
        f1_r_v3 = (2.0 * float(tp_v3[idx].sum()) / denom_v3) if denom_v3 > 0.0 else 0.0

        diferencias[i] = f1_r_v3 - f1_r_v2

    return resumir_distribucion_bootstrap(
        diferencias=diferencias,
        diff_observada=diff_obs,
        n_bloques=n_bloques,
        unidad_bloque=unidad_bloque,
        metrica="f1_score",
        direccion="v3 - v2 (positivo favorece a v3)",
    )


def bootstrap_bloques_pr_auc(
    y_real: np.ndarray,
    prob_v2: np.ndarray,
    prob_v3: np.ndarray,
    bloques: np.ndarray,
    unidad_bloque: str,
    n_bootstrap: int = N_BOOTSTRAP_PR_AUC,
    semilla: int = config.RANDOM_SEED,
) -> dict:
    """
    Ejecuta bootstrap por bloques para la diferencia de PR-AUC (v3 - v2) en P2.
    Cada réplica asigna a cada fila un peso igual al número de veces que su bloque
    (día o estación) resultó seleccionado con reemplazo, lo cual equivale de forma
    exacta a remuestrear bloques completos.
    """
    y_arr = np.asarray(y_real, dtype="int64")
    p2_arr = np.asarray(prob_v2, dtype="float64")
    p3_arr = np.asarray(prob_v3, dtype="float64")

    codigos_bloque, bloques_unicos = pd.factorize(bloques, sort=True)
    n_bloques = len(bloques_unicos)

    pr_obs_v2 = float(average_precision_score(y_arr, p2_arr))
    pr_obs_v3 = float(average_precision_score(y_arr, p3_arr))
    diff_obs = pr_obs_v3 - pr_obs_v2

    rng = np.random.default_rng(semilla)
    diferencias = np.empty(n_bootstrap, dtype="float64")

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_bloques, size=n_bloques)
        conteos_bloque = np.bincount(idx, minlength=n_bloques)
        pesos_fila = conteos_bloque[codigos_bloque]

        pr_r_v2 = float(average_precision_score(y_arr, p2_arr, sample_weight=pesos_fila))
        pr_r_v3 = float(average_precision_score(y_arr, p3_arr, sample_weight=pesos_fila))
        diferencias[i] = pr_r_v3 - pr_r_v2

    return resumir_distribucion_bootstrap(
        diferencias=diferencias,
        diff_observada=diff_obs,
        n_bloques=n_bloques,
        unidad_bloque=unidad_bloque,
        metrica="pr_auc",
        direccion="v3 - v2 (positivo favorece a v3)",
    )


def bootstrap_estaciones_silhouette(
    X_norm: np.ndarray,
    clusters_v2: np.ndarray,
    clusters_v3: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_SILHOUETTE,
    semilla: int = config.RANDOM_SEED,
) -> dict:
    """
    Ejecuta bootstrap por estaciones para la diferencia de Silhouette Score (v3 - v2)
    en P3. Se reporta como métrica complementaria junto con la distancia media al
    centroide (la cual no depende de distancias entre pares duplicados al remuestrear).
    """
    n_estaciones = len(X_norm)
    sil_obs_v2 = float(silhouette_score(X_norm, clusters_v2))
    sil_obs_v3 = float(silhouette_score(X_norm, clusters_v3))
    diff_obs = sil_obs_v3 - sil_obs_v2

    rng = np.random.default_rng(semilla)
    diferencias_validas = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_estaciones, size=n_estaciones)
        c2_r = clusters_v2[idx]
        c3_r = clusters_v3[idx]
        # Verificamos que existan al menos 2 grupos distintos en la réplica
        if len(np.unique(c2_r)) < 2 or len(np.unique(c3_r)) < 2:
            continue
        sil_r_v2 = float(silhouette_score(X_norm[idx], c2_r))
        sil_r_v3 = float(silhouette_score(X_norm[idx], c3_r))
        diferencias_validas.append(sil_r_v3 - sil_r_v2)

    return resumir_distribucion_bootstrap(
        diferencias=np.asarray(diferencias_validas, dtype="float64"),
        diff_observada=diff_obs,
        n_bloques=n_estaciones,
        unidad_bloque="estaciones",
        metrica="silhouette_score",
        direccion="v3 - v2 (positivo favorece a v3)",
    )


# ==============================================================================
# 5. EJECUTAR EL A/B TESTING SOBRE EL CONJUNTO DE EVALUACIÓN DE OCTUBRE 2026
# ==============================================================================
def evaluar_ab_testing_octubre_2026() -> dict:
    """
    Compara la Variante A (v2: modelo entrenado hasta Dic 2025) contra la Variante B
    (v3: modelo reentrenado hasta Sep 2026 con nomenclatura 2026_10) sobre todo el
    periodo de producción Octubre 2026 para P1, P2 y P3, estimando la incertidumbre
    mediante bootstrap por bloques (por días y por estaciones).
    """
    obtener_o_entrenar_version_v3()
    df_p1_p2_oct, df_p3_oct, manifiesto_10 = drift_report.construir_features_octubre_2026()

    cols_x_p1_p2 = drift_report.COLUMNAS_FEATURES_P1_P2
    cols_x_p3 = drift_report.COLUMNAS_FEATURES_P3

    X_eval_p1_p2 = df_p1_p2_oct[cols_x_p1_p2].astype("float64")
    y_eval_p1 = df_p1_p2_oct["viajes_iniciados"].astype("float64").to_numpy()
    y_eval_p2 = df_p1_p2_oct["en_riesgo"].astype("int64").to_numpy()
    mascara_pico = (df_p1_p2_oct["es_hora_pico"] == 1).to_numpy()

    # Bloques reales de dependencia temporal (31 días) y espacial (469 estaciones)
    bloques_dias = df_p1_p2_oct["Fecha_Hora"].dt.strftime("%Y-%m-%d").to_numpy()
    bloques_estaciones = df_p1_p2_oct["Estacion_Id"].astype("int64").to_numpy()
    n_dias_unicos = int(pd.Series(bloques_dias).nunique())
    n_estaciones_p1_p2 = int(pd.Series(bloques_estaciones).nunique())

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P1 (REGRESIÓN DE DEMANDA HORARIA)
    # --------------------------------------------------------------------------
    print("\nEvaluando A/B Testing de P1 con bootstrap por días y por estaciones...")
    m_p1_v2 = cargar_version_modelo_registry(
        config.MODEL_NAME_P1, config.AB_VARIANT_A_VERSION
    )
    m_p1_v3 = cargar_version_modelo_registry(
        config.MODEL_NAME_P1, config.AB_VARIANT_B_VERSION
    )

    pred_p1_v2 = np.clip(m_p1_v2.predict(X_eval_p1_p2), 0.0, None)
    pred_p1_v3 = np.clip(m_p1_v3.predict(X_eval_p1_p2), 0.0, None)

    err_abs_v2 = np.abs(y_eval_p1 - pred_p1_v2)
    err_abs_v3 = np.abs(y_eval_p1 - pred_p1_v3)

    boot_p1_dias = bootstrap_bloques_media_error(
        error_v2=err_abs_v2,
        error_v3=err_abs_v3,
        bloques=bloques_dias,
        unidad_bloque="dias",
        metrica="mae",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED,
    )
    boot_p1_estaciones = bootstrap_bloques_media_error(
        error_v2=err_abs_v2,
        error_v3=err_abs_v3,
        bloques=bloques_estaciones,
        unidad_bloque="estaciones",
        metrica="mae",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 1,
    )

    p1_v2_metrics = {
        "mae": round(float(mean_absolute_error(y_eval_p1, pred_p1_v2)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_eval_p1, pred_p1_v2))), 4),
        "r2": round(float(r2_score(y_eval_p1, pred_p1_v2)), 4),
        "peak_hour_mae": round(
            float(mean_absolute_error(y_eval_p1[mascara_pico], pred_p1_v2[mascara_pico])),
            4,
        ),
        "mean_bias_error": round(float(np.mean(pred_p1_v2 - y_eval_p1)), 4),
        "pred_mean": round(float(np.mean(pred_p1_v2)), 4),
    }
    p1_v3_metrics = {
        "mae": round(float(mean_absolute_error(y_eval_p1, pred_p1_v3)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_eval_p1, pred_p1_v3))), 4),
        "r2": round(float(r2_score(y_eval_p1, pred_p1_v3)), 4),
        "peak_hour_mae": round(
            float(mean_absolute_error(y_eval_p1[mascara_pico], pred_p1_v3[mascara_pico])),
            4,
        ),
        "mean_bias_error": round(float(np.mean(pred_p1_v3 - y_eval_p1)), 4),
        "pred_mean": round(float(np.mean(pred_p1_v3)), 4),
    }

    # Promovemos v3 si mejora el MAE y la diferencia es estadísticamente significativa
    # tanto al remuestrear por días como por estaciones (IC 95% estrictamente < 0)
    p1_sig_ambos = bool(
        boot_p1_dias["significant_ci95"] and boot_p1_estaciones["significant_ci95"]
    )
    ganador_p1 = (
        "v3"
        if (p1_v3_metrics["mae"] < p1_v2_metrics["mae"] and p1_sig_ambos)
        else "v2"
    )

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P2 (CLASIFICACIÓN DE RIESGO DE DESBALANCE >= 30%)
    # --------------------------------------------------------------------------
    print("Evaluando A/B Testing de P2 con bootstrap por días y por estaciones...")
    m_p2_v2 = cargar_version_modelo_registry(
        config.MODEL_NAME_P2, config.AB_VARIANT_A_VERSION
    )
    m_p2_v3 = cargar_version_modelo_registry(
        config.MODEL_NAME_P2, config.AB_VARIANT_B_VERSION
    )

    pred_p2_v2 = m_p2_v2.predict(X_eval_p1_p2)
    pred_p2_v3 = m_p2_v3.predict(X_eval_p1_p2)
    prob_p2_v2 = m_p2_v2.predict_proba(X_eval_p1_p2)[:, 1]
    prob_p2_v3 = m_p2_v3.predict_proba(X_eval_p1_p2)[:, 1]

    brier_fila_v2 = (y_eval_p2 - prob_p2_v2) ** 2
    brier_fila_v3 = (y_eval_p2 - prob_p2_v3) ** 2

    boot_p2_f1_dias = bootstrap_bloques_f1(
        y_real=y_eval_p2,
        pred_v2=pred_p2_v2,
        pred_v3=pred_p2_v3,
        bloques=bloques_dias,
        unidad_bloque="dias",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 2,
    )
    boot_p2_f1_estaciones = bootstrap_bloques_f1(
        y_real=y_eval_p2,
        pred_v2=pred_p2_v2,
        pred_v3=pred_p2_v3,
        bloques=bloques_estaciones,
        unidad_bloque="estaciones",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 3,
    )
    boot_p2_brier_dias = bootstrap_bloques_media_error(
        error_v2=brier_fila_v2,
        error_v3=brier_fila_v3,
        bloques=bloques_dias,
        unidad_bloque="dias",
        metrica="brier_score",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 4,
    )
    boot_p2_brier_estaciones = bootstrap_bloques_media_error(
        error_v2=brier_fila_v2,
        error_v3=brier_fila_v3,
        bloques=bloques_estaciones,
        unidad_bloque="estaciones",
        metrica="brier_score",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 5,
    )
    boot_p2_prauc_dias = bootstrap_bloques_pr_auc(
        y_real=y_eval_p2,
        prob_v2=prob_p2_v2,
        prob_v3=prob_p2_v3,
        bloques=bloques_dias,
        unidad_bloque="dias",
        n_bootstrap=N_BOOTSTRAP_PR_AUC,
        semilla=config.RANDOM_SEED + 6,
    )
    boot_p2_prauc_estaciones = bootstrap_bloques_pr_auc(
        y_real=y_eval_p2,
        prob_v2=prob_p2_v2,
        prob_v3=prob_p2_v3,
        bloques=bloques_estaciones,
        unidad_bloque="estaciones",
        n_bootstrap=N_BOOTSTRAP_PR_AUC,
        semilla=config.RANDOM_SEED + 7,
    )

    p2_v2_metrics = {
        "f1": round(float(f1_score(y_eval_p2, pred_p2_v2, zero_division=0)), 4),
        "precision": round(float(precision_score(y_eval_p2, pred_p2_v2, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval_p2, pred_p2_v2, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_eval_p2, prob_p2_v2)), 4),
        "pr_auc": round(float(average_precision_score(y_eval_p2, prob_p2_v2)), 4),
        "brier_score": round(float(brier_score_loss(y_eval_p2, prob_p2_v2)), 4),
    }
    p2_v3_metrics = {
        "f1": round(float(f1_score(y_eval_p2, pred_p2_v3, zero_division=0)), 4),
        "precision": round(float(precision_score(y_eval_p2, pred_p2_v3, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval_p2, pred_p2_v3, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_eval_p2, prob_p2_v3)), 4),
        "pr_auc": round(float(average_precision_score(y_eval_p2, prob_p2_v3)), 4),
        "brier_score": round(float(brier_score_loss(y_eval_p2, prob_p2_v3)), 4),
    }

    # Promovemos v3 en P2 únicamente si mejora el F1 operativo al umbral 0.50 con
    # significancia estadística; de lo contrario se protege v2 como @champion
    ganador_p2 = (
        "v3"
        if (
            p2_v3_metrics["f1"] > p2_v2_metrics["f1"]
            and boot_p2_f1_dias["significant_ci95"]
            and boot_p2_f1_estaciones["significant_ci95"]
        )
        else "v2"
    )

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P3 (CLUSTERING DE ESTACIONES EN OCTUBRE 2026)
    # --------------------------------------------------------------------------
    print("Evaluando A/B Testing de P3 con bootstrap por estaciones...")
    m_p3_v2 = cargar_version_modelo_registry(
        config.MODEL_NAME_P3, config.AB_VARIANT_A_VERSION
    )
    m_p3_v3 = cargar_version_modelo_registry(
        config.MODEL_NAME_P3, config.AB_VARIANT_B_VERSION
    )

    X_eval_p3 = df_p3_oct[cols_x_p3].astype("float64")
    X_eval_p3_norm = StandardScaler().fit_transform(X_eval_p3)
    bloques_estaciones_p3 = df_p3_oct["Estacion_Id"].astype("int64").to_numpy()

    clusters_v2 = m_p3_v2.predict(X_eval_p3)
    clusters_v3 = m_p3_v3.predict(X_eval_p3)

    # Distancia euclidiana de cada estación a su centroide asignado
    X_trans_v2 = m_p3_v2.named_steps["scaler"].transform(X_eval_p3)
    centroides_v2 = m_p3_v2.named_steps["clusterer"].cluster_centers_
    dist_centroide_v2 = np.linalg.norm(X_trans_v2 - centroides_v2[clusters_v2], axis=1)

    X_trans_v3 = m_p3_v3.named_steps["scaler"].transform(X_eval_p3)
    centroides_v3 = m_p3_v3.named_steps["clusterer"].cluster_centers_
    dist_centroide_v3 = np.linalg.norm(X_trans_v3 - centroides_v3[clusters_v3], axis=1)

    boot_p3_distancia_estaciones = bootstrap_bloques_media_error(
        error_v2=dist_centroide_v2,
        error_v3=dist_centroide_v3,
        bloques=bloques_estaciones_p3,
        unidad_bloque="estaciones",
        metrica="mean_centroid_distance",
        n_bootstrap=N_BOOTSTRAP_BLOQUES,
        semilla=config.RANDOM_SEED + 8,
    )
    boot_p3_silhouette_estaciones = bootstrap_estaciones_silhouette(
        X_norm=X_eval_p3_norm,
        clusters_v2=clusters_v2,
        clusters_v3=clusters_v3,
        n_bootstrap=N_BOOTSTRAP_SILHOUETTE,
        semilla=config.RANDOM_SEED + 9,
    )

    prop_min_v2 = float(pd.Series(clusters_v2).value_counts().min() / len(X_eval_p3))
    prop_min_v3 = float(pd.Series(clusters_v3).value_counts().min() / len(X_eval_p3))

    p3_v2_metrics = {
        "silhouette_oct2026": round(float(silhouette_score(X_eval_p3_norm, clusters_v2)), 4),
        "davies_bouldin_oct2026": round(
            float(davies_bouldin_score(X_eval_p3_norm, clusters_v2)), 4
        ),
        "mean_centroid_distance": round(float(np.mean(dist_centroide_v2)), 4),
        "min_cluster_share": round(prop_min_v2, 4),
        "cluster_counts": {
            str(k): int(v)
            for k, v in pd.Series(clusters_v2).value_counts().sort_index().items()
        },
    }
    p3_v3_metrics = {
        "silhouette_oct2026": round(float(silhouette_score(X_eval_p3_norm, clusters_v3)), 4),
        "davies_bouldin_oct2026": round(
            float(davies_bouldin_score(X_eval_p3_norm, clusters_v3)), 4
        ),
        "mean_centroid_distance": round(float(np.mean(dist_centroide_v3)), 4),
        "min_cluster_share": round(prop_min_v3, 4),
        "cluster_counts": {
            str(k): int(v)
            for k, v in pd.Series(clusters_v3).value_counts().sort_index().items()
        },
    }

    # En P3 promovemos v3 si mejora la cohesión (Silhouette) y reduce significativamente
    # la distancia al centroide al remuestrear por estaciones
    ganador_p3 = (
        "v3"
        if (
            p3_v3_metrics["silhouette_oct2026"] > p3_v2_metrics["silhouette_oct2026"]
            and boot_p3_distancia_estaciones["significant_ci95"]
            and boot_p3_distancia_estaciones["observed_diff"] < 0.0
        )
        else "v2"
    )

    resultados_ab = {
        "evaluation_period": "Octubre 2026 (2026-10-01 a 2026-10-31)",
        "evaluation_data_version": manifiesto_10["data_version_short"],
        "hourly_observations_evaluated": int(len(X_eval_p1_p2)),
        "unique_days_evaluated": n_dias_unicos,
        "active_stations_evaluated_p1_p2": n_estaciones_p1_p2,
        "active_stations_evaluated_p3": int(len(X_eval_p3)),
        "statistical_method": {
            "name": "Block Bootstrap Resampling (Days and Stations)",
            "random_seed": config.RANDOM_SEED,
            "n_bootstrap_primary": N_BOOTSTRAP_BLOQUES,
            "n_bootstrap_pr_auc": N_BOOTSTRAP_PR_AUC,
            "n_bootstrap_silhouette": N_BOOTSTRAP_SILHOUETTE,
            "confidence_level": 0.95,
            "justification": (
                "Las observaciones estacion x hora presentan autocorrelacion temporal "
                "(rezagos lag_1, lag_2, lag_24 y media movil 3h) y dependencia espacial "
                "compartida por red y estacion. El remuestreo bootstrap por bloques "
                "(31 dias y 469 estaciones) evita inflar artificialmente el tamano muestral "
                "efectivo y estima intervalos de confianza al 95% realistas directamente "
                "sobre las metricas operativas de negocio."
            ),
        },
        "models": {
            "p1": {
                "model_name": config.MODEL_NAME_P1,
                "real_target_mean_oct2026": round(float(np.mean(y_eval_p1)), 4),
                "variant_A_v2": p1_v2_metrics,
                "variant_B_v3": p1_v3_metrics,
                "block_bootstrap": {
                    "mae_by_day": boot_p1_dias,
                    "mae_by_station": boot_p1_estaciones,
                },
                "statistically_significant_ci95": p1_sig_ambos,
                "selected_winner": ganador_p1,
            },
            "p2": {
                "model_name": config.MODEL_NAME_P2,
                "real_positive_rate_oct2026": round(float(np.mean(y_eval_p2)), 4),
                "variant_A_v2": p2_v2_metrics,
                "variant_B_v3": p2_v3_metrics,
                "block_bootstrap": {
                    "f1_by_day": boot_p2_f1_dias,
                    "f1_by_station": boot_p2_f1_estaciones,
                    "brier_by_day": boot_p2_brier_dias,
                    "brier_by_station": boot_p2_brier_estaciones,
                    "pr_auc_by_day": boot_p2_prauc_dias,
                    "pr_auc_by_station": boot_p2_prauc_estaciones,
                },
                "statistically_significant_f1_ci95": bool(
                    boot_p2_f1_dias["significant_ci95"]
                    and boot_p2_f1_estaciones["significant_ci95"]
                ),
                "selected_winner": ganador_p2,
            },
            "p3": {
                "model_name": config.MODEL_NAME_P3,
                "variant_A_v2": p3_v2_metrics,
                "variant_B_v3": p3_v3_metrics,
                "block_bootstrap": {
                    "centroid_distance_by_station": boot_p3_distancia_estaciones,
                    "silhouette_by_station": boot_p3_silhouette_estaciones,
                },
                "statistically_significant_ci95": bool(
                    boot_p3_distancia_estaciones["significant_ci95"]
                ),
                "selected_winner": ganador_p3,
            },
        },
    }

    ruta_fig_ab = graficar_resultados_ab_testing(resultados_ab)
    resultados_ab["figure_path"] = str(ruta_fig_ab.relative_to(config.PROJ_ROOT))

    config.MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.AB_TEST_SUMMARY_JSON, "w", encoding="utf-8") as f_out:
        json.dump(resultados_ab, f_out, indent=4, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("RESULTADOS DEL A/B TESTING CON BOOTSTRAP POR BLOQUES (OCTUBRE 2026)")
    print("=" * 90)
    print(
        f"P1 y P2: {len(X_eval_p1_p2):,} obs. horarias | {n_dias_unicos} días | "
        f"{n_estaciones_p1_p2} estaciones"
    )
    print(f"P3:      {len(X_eval_p3):,} estaciones activas")

    print("\n1. P1 - Regresión de Demanda Horaria (mibici-equipo35-p1-demanda):")
    print(f"   * Variante A (v2): {p1_v2_metrics}")
    print(f"   * Variante B (v3): {p1_v3_metrics}")
    print(
        f"   * Bootstrap MAE por DÍAS (31):       diff={boot_p1_dias['observed_diff']:+.4f} "
        f"IC95%=[{boot_p1_dias['ci95_lower']:+.4f}, {boot_p1_dias['ci95_upper']:+.4f}] "
        f"p={boot_p1_dias['p_value']:.4f} sig={boot_p1_dias['significant_ci95']}"
    )
    print(
        f"   * Bootstrap MAE por ESTACIONES (469): diff={boot_p1_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p1_estaciones['ci95_lower']:+.4f}, {boot_p1_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p1_estaciones['p_value']:.4f} sig={boot_p1_estaciones['significant_ci95']}"
    )
    print(f"   -> Ganador seleccionado: {ganador_p1}")

    print("\n2. P2 - Clasificación de Riesgo Horario (mibici-equipo35-p2-riesgo):")
    print(f"   * Variante A (v2): {p2_v2_metrics}")
    print(f"   * Variante B (v3): {p2_v3_metrics}")
    print(
        f"   * Bootstrap F1 por DÍAS (31):         diff={boot_p2_f1_dias['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_f1_dias['ci95_lower']:+.4f}, {boot_p2_f1_dias['ci95_upper']:+.4f}] "
        f"p={boot_p2_f1_dias['p_value']:.4f} sig={boot_p2_f1_dias['significant_ci95']}"
    )
    print(
        f"   * Bootstrap F1 por ESTACIONES (469):   diff={boot_p2_f1_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_f1_estaciones['ci95_lower']:+.4f}, {boot_p2_f1_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p2_f1_estaciones['p_value']:.4f} sig={boot_p2_f1_estaciones['significant_ci95']}"
    )
    print(
        f"   * Bootstrap Brier por DÍAS (31):       diff={boot_p2_brier_dias['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_brier_dias['ci95_lower']:+.4f}, {boot_p2_brier_dias['ci95_upper']:+.4f}] "
        f"p={boot_p2_brier_dias['p_value']:.4f} sig={boot_p2_brier_dias['significant_ci95']}"
    )
    print(
        f"   * Bootstrap Brier por ESTACIONES (469):diff={boot_p2_brier_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_brier_estaciones['ci95_lower']:+.4f}, {boot_p2_brier_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p2_brier_estaciones['p_value']:.4f} sig={boot_p2_brier_estaciones['significant_ci95']}"
    )
    print(
        f"   * Bootstrap PR-AUC por DÍAS (31):      diff={boot_p2_prauc_dias['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_prauc_dias['ci95_lower']:+.4f}, {boot_p2_prauc_dias['ci95_upper']:+.4f}] "
        f"p={boot_p2_prauc_dias['p_value']:.4f} sig={boot_p2_prauc_dias['significant_ci95']}"
    )
    print(
        f"   * Bootstrap PR-AUC por ESTACIONES (469):diff={boot_p2_prauc_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p2_prauc_estaciones['ci95_lower']:+.4f}, {boot_p2_prauc_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p2_prauc_estaciones['p_value']:.4f} sig={boot_p2_prauc_estaciones['significant_ci95']}"
    )
    print(f"   -> Ganador seleccionado: {ganador_p2}")

    print("\n3. P3 - Clustering de Perfiles de Estaciones (mibici-equipo35-p3-grupos):")
    print(f"   * Variante A (v2): {p3_v2_metrics}")
    print(f"   * Variante B (v3): {p3_v3_metrics}")
    print(
        f"   * Bootstrap Dist. Centroide (469 est.): diff={boot_p3_distancia_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p3_distancia_estaciones['ci95_lower']:+.4f}, {boot_p3_distancia_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p3_distancia_estaciones['p_value']:.4f} sig={boot_p3_distancia_estaciones['significant_ci95']}"
    )
    print(
        f"   * Bootstrap Silhouette (469 est.):      diff={boot_p3_silhouette_estaciones['observed_diff']:+.4f} "
        f"IC95%=[{boot_p3_silhouette_estaciones['ci95_lower']:+.4f}, {boot_p3_silhouette_estaciones['ci95_upper']:+.4f}] "
        f"p={boot_p3_silhouette_estaciones['p_value']:.4f} sig={boot_p3_silhouette_estaciones['significant_ci95']}"
    )
    print(f"   -> Ganador seleccionado: {ganador_p3}")
    print("=" * 90)

    # Actualizamos aliases (@champion / @challenger) y linaje de v3 en MLflow Registry
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        cliente = MlflowClient()
        for preg, nom_mod in [
            ("p1", config.MODEL_NAME_P1),
            ("p2", config.MODEL_NAME_P2),
            ("p3", config.MODEL_NAME_P3),
        ]:
            ganador = resultados_ab["models"][preg]["selected_winner"]
            ver_champ = "3" if ganador == "v3" else "2"
            ver_chall = "2" if ganador == "v3" else "3"
            cliente.set_registered_model_alias(
                nom_mod, config.PRODUCTION_ALIAS, ver_champ
            )
            cliente.set_registered_model_alias(
                nom_mod, config.CHALLENGER_ALIAS, ver_chall
            )

            # Enriquecemos la descripción y tags de v3 con su linaje completo y estado
            mv3 = cliente.get_model_version(nom_mod, "3")
            estado_v3 = (
                "Modelo Champion en Producción (@champion)"
                if ver_champ == "3"
                else "Modelo Challenger Retador (@challenger; se mantiene v2 en producción)"
            )
            git_tag_v3 = mv3.tags.get("git_commit", "N/A")
            data_tag_v3 = mv3.tags.get("data_version", "128979137b46+3542d1d567e1")
            cliente.update_model_version(
                name=nom_mod,
                version="3",
                description=(
                    f"{estado_v3}. Versión v3 (Variante B - Continuous Training) "
                    "reentrenada hasta Sep 2026 con nomenclatura_2026_10.csv y validada "
                    "mediante A/B Testing con bootstrap por bloques (días y estaciones). "
                    f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} "
                    f"| Run ID: {mv3.run_id} | Git Commit: {git_tag_v3} "
                    f"| Data Version: {data_tag_v3}."
                ),
            )
            cliente.set_model_version_tag(
                nom_mod, "3", "ab_test_method", "block_bootstrap_days_and_stations"
            )
            cliente.set_model_version_tag(
                nom_mod,
                "3",
                "production_status",
                "champion" if ver_champ == "3" else "challenger",
            )

        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="AB_Testing_BlockBootstrap_Oct2026"):
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "stage_of_work": "ab_testing_block_bootstrap",
                    "evaluation_period": "2026-10",
                    "statistical_method": "block_bootstrap_days_and_stations",
                    "winner_p1": ganador_p1,
                    "winner_p2": ganador_p2,
                    "winner_p3": ganador_p3,
                }
            )
            mlflow.log_metrics(
                {
                    "p1_v2_mae": p1_v2_metrics["mae"],
                    "p1_v3_mae": p1_v3_metrics["mae"],
                    "p1_boot_days_p_value": boot_p1_dias["p_value"],
                    "p1_boot_stations_p_value": boot_p1_estaciones["p_value"],
                    "p2_v2_f1": p2_v2_metrics["f1"],
                    "p2_v3_f1": p2_v3_metrics["f1"],
                    "p2_boot_f1_days_p_value": boot_p2_f1_dias["p_value"],
                    "p2_boot_f1_stations_p_value": boot_p2_f1_estaciones["p_value"],
                    "p2_boot_brier_days_p_value": boot_p2_brier_dias["p_value"],
                    "p2_boot_brier_stations_p_value": boot_p2_brier_estaciones["p_value"],
                    "p2_boot_prauc_days_p_value": boot_p2_prauc_dias["p_value"],
                    "p2_boot_prauc_stations_p_value": boot_p2_prauc_estaciones["p_value"],
                    "p3_v2_silhouette": p3_v2_metrics["silhouette_oct2026"],
                    "p3_v3_silhouette": p3_v3_metrics["silhouette_oct2026"],
                    "p3_boot_dist_stations_p_value": boot_p3_distancia_estaciones["p_value"],
                    "p3_boot_sil_stations_p_value": boot_p3_silhouette_estaciones["p_value"],
                }
            )
            mlflow.log_artifact(str(config.AB_TEST_SUMMARY_JSON), artifact_path="monitoring")
            mlflow.log_artifact(str(ruta_fig_ab), artifact_path="plots")
    except Exception as exc:
        print(f"Aviso: No se pudieron actualizar aliases/metadatos en MLflow ({exc}).")

    return resultados_ab


# ==============================================================================
# 6. GENERAR GRÁFICA COMPARATIVA DEL A/B TESTING + INTERVALOS BOOTSTRAP AL 95%
# ==============================================================================
def graficar_resultados_ab_testing(resultados_ab: dict) -> Path:
    """
    Genera una figura de 2 filas x 3 columnas:
      - Fila superior: comparación de métricas de Variante A (v2) vs. Variante B (v3)
        para P1, P2 y P3 sobre Octubre 2026.
      - Fila inferior: diferencias observadas (v3 - v2) con sus intervalos de confianza
        al 95% obtenidos mediante bootstrap por bloques (días y estaciones).
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = config.FIGURES_DIR / "ab_testing_comparacion_modelos.png"

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), dpi=150)
    ancho = 0.34

    # --- FILA SUPERIOR: MÉTRICAS COMPARADAS v2 vs v3 ---
    # Panel (0, 0): P1 Regresión
    ax1 = axes[0, 0]
    m1_a = resultados_ab["models"]["p1"]["variant_A_v2"]
    m1_b = resultados_ab["models"]["p1"]["variant_B_v3"]
    etiquetas_p1 = ["MAE\nGlobal", "MAE\nHora Pico", "RMSE", "|Sesgo|\nMedio"]
    vals_p1_a = [
        m1_a["mae"],
        m1_a["peak_hour_mae"],
        m1_a["rmse"],
        abs(m1_a["mean_bias_error"]),
    ]
    vals_p1_b = [
        m1_b["mae"],
        m1_b["peak_hour_mae"],
        m1_b["rmse"],
        abs(m1_b["mean_bias_error"]),
    ]
    x1 = np.arange(len(etiquetas_p1))
    ax1.bar(
        x1 - ancho / 2,
        vals_p1_a,
        width=ancho,
        label="Variante A (v2)",
        color="#2b8cbe",
        edgecolor="black",
    )
    ax1.bar(
        x1 + ancho / 2,
        vals_p1_b,
        width=ancho,
        label="Variante B (v3)",
        color="#2ca25f",
        edgecolor="black",
    )
    for idx, (va, vb) in enumerate(zip(vals_p1_a, vals_p1_b)):
        ax1.text(idx - ancho / 2, va + 0.05, f"{va:.2f}", ha="center", fontsize=7.8)
        ax1.text(
            idx + ancho / 2,
            vb + 0.05,
            f"{vb:.2f}",
            ha="center",
            fontsize=7.8,
            fontweight="bold",
        )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(etiquetas_p1, fontsize=8.5)
    ax1.set_title("P1 Demanda (Menor = Mejor)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(vals_p1_a + vals_p1_b) + 0.45)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Panel (0, 1): P2 Clasificación
    ax2 = axes[0, 1]
    m2_a = resultados_ab["models"]["p2"]["variant_A_v2"]
    m2_b = resultados_ab["models"]["p2"]["variant_B_v3"]
    etiquetas_p2 = ["F1-Score", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    vals_p2_a = [
        m2_a["f1"],
        m2_a["precision"],
        m2_a["recall"],
        m2_a["roc_auc"],
        m2_a["pr_auc"],
    ]
    vals_p2_b = [
        m2_b["f1"],
        m2_b["precision"],
        m2_b["recall"],
        m2_b["roc_auc"],
        m2_b["pr_auc"],
    ]
    x2 = np.arange(len(etiquetas_p2))
    ax2.bar(
        x2 - ancho / 2,
        vals_p2_a,
        width=ancho,
        label="Variante A (v2)",
        color="#2b8cbe",
        edgecolor="black",
    )
    ax2.bar(
        x2 + ancho / 2,
        vals_p2_b,
        width=ancho,
        label="Variante B (v3)",
        color="#2ca25f",
        edgecolor="black",
    )
    for idx, (va, vb) in enumerate(zip(vals_p2_a, vals_p2_b)):
        ax2.text(
            idx - ancho / 2 - 0.03, va + 0.018, f"{va:.3f}", ha="center", fontsize=6.8
        )
        ax2.text(
            idx + ancho / 2 + 0.03,
            vb + 0.038,
            f"{vb:.3f}",
            ha="center",
            fontsize=6.8,
            fontweight="bold",
        )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(etiquetas_p2, fontsize=8.2)
    ax2.set_title("P2 Riesgo (Mayor = Mejor)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.08)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panel (0, 2): P3 Clustering
    ax3 = axes[0, 2]
    m3_a = resultados_ab["models"]["p3"]["variant_A_v2"]
    m3_b = resultados_ab["models"]["p3"]["variant_B_v3"]
    etiquetas_p3 = [
        "Silhouette\n(Mayor mejor)",
        "Dist. Centroide\n(Menor mejor)",
        "Davies-Bouldin\n(Menor mejor)",
    ]
    vals_p3_a = [
        m3_a["silhouette_oct2026"],
        m3_a["mean_centroid_distance"],
        m3_a["davies_bouldin_oct2026"],
    ]
    vals_p3_b = [
        m3_b["silhouette_oct2026"],
        m3_b["mean_centroid_distance"],
        m3_b["davies_bouldin_oct2026"],
    ]
    x3 = np.arange(len(etiquetas_p3))
    ax3.bar(
        x3 - ancho / 2,
        vals_p3_a,
        width=ancho,
        label="Variante A (v2)",
        color="#2b8cbe",
        edgecolor="black",
    )
    ax3.bar(
        x3 + ancho / 2,
        vals_p3_b,
        width=ancho,
        label="Variante B (v3)",
        color="#2ca25f",
        edgecolor="black",
    )
    for idx, (va, vb) in enumerate(zip(vals_p3_a, vals_p3_b)):
        ax3.text(idx - ancho / 2, va + 0.04, f"{va:.3f}", ha="center", fontsize=7.8)
        ax3.text(
            idx + ancho / 2,
            vb + 0.04,
            f"{vb:.3f}",
            ha="center",
            fontsize=7.8,
            fontweight="bold",
        )
    ax3.set_xticks(x3)
    ax3.set_xticklabels(etiquetas_p3, fontsize=8.2)
    ax3.set_title("P3 Perfiles Estaciones (Oct 2026)", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(vals_p3_a + vals_p3_b) + 0.35)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # --- FILA INFERIOR: INTERVALOS DE CONFIANZA AL 95% (BOOTSTRAP POR BLOQUES) ---
    def _dibujar_ic_bootstrap(ax, items_ic: list[tuple[str, dict, bool]], titulo: str, nota_eje: str):
        """
        Dibuja barras de error horizontales para las diferencias (v3 - v2) y sus IC 95%.
        items_ic: lista de (etiqueta, diccionario_bootstrap, menor_es_mejor).
        """
        y_pos = np.arange(len(items_ic))
        for idx, (etiqueta, d_boot, menor_mejor) in enumerate(items_ic):
            diff = d_boot["observed_diff"]
            err_izq = max(0.0, diff - d_boot["ci95_lower"])
            err_der = max(0.0, d_boot["ci95_upper"] - diff)
            favorece_v3 = (diff < 0) if menor_mejor else (diff > 0)
            if not d_boot["significant_ci95"]:
                color_pt = "#756bb1"  # No significativo (cruza 0)
            elif favorece_v3:
                color_pt = "#2ca25f"  # Significativo a favor de v3
            else:
                color_pt = "#de2d26"  # Significativo a favor de v2

            ax.errorbar(
                diff,
                idx,
                xerr=[[err_izq], [err_der]],
                fmt="o",
                color=color_pt,
                ecolor=color_pt,
                elinewidth=2.0,
                capsize=4,
                markersize=6.5,
            )
            ax.annotate(
                f"{diff:+.4f} (p={d_boot['p_value']:.3f})",
                xy=(diff, idx),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=7.3,
                color="#222222",
            )

        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([it[0] for it in items_ic], fontsize=8.0)
        ax.set_ylim(-0.6, len(items_ic) - 0.2)
        ax.margins(x=0.22)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_title(titulo, fontsize=9.5, fontweight="bold")
        ax.set_xlabel(nota_eje, fontsize=7.8)
        ax.grid(axis="x", alpha=0.3)

    b1 = resultados_ab["models"]["p1"]["block_bootstrap"]
    _dibujar_ic_bootstrap(
        axes[1, 0],
        [
            ("MAE\n(31 días)", b1["mae_by_day"], True),
            ("MAE\n(469 est.)", b1["mae_by_station"], True),
        ],
        "P1 Bootstrap IC 95%: Δ MAE (v3 - v2)",
        "Δ < 0 favorece a v3 | Verde=v3, Rojo=v2, Morado=NS",
    )

    b2 = resultados_ab["models"]["p2"]["block_bootstrap"]
    _dibujar_ic_bootstrap(
        axes[1, 1],
        [
            ("F1 (31 días)", b2["f1_by_day"], False),
            ("F1 (469 est.)", b2["f1_by_station"], False),
            ("Brier (31 días)", b2["brier_by_day"], True),
            ("Brier (469 est.)", b2["brier_by_station"], True),
            ("PR-AUC (31 días)", b2["pr_auc_by_day"], False),
            ("PR-AUC (469 est.)", b2["pr_auc_by_station"], False),
        ],
        "P2 Bootstrap IC 95%: Δ Métricas (v3 - v2)",
        "F1/PR-AUC > 0 y Brier < 0 favorecen a v3",
    )

    b3 = resultados_ab["models"]["p3"]["block_bootstrap"]
    _dibujar_ic_bootstrap(
        axes[1, 2],
        [
            ("Dist. Centroide\n(469 est.)", b3["centroid_distance_by_station"], True),
            ("Silhouette\n(469 est.)", b3["silhouette_by_station"], False),
        ],
        "P3 Bootstrap IC 95%: Δ Clustering (v3 - v2)",
        "Dist. < 0 y Silhouette > 0 favorecen a v3",
    )

    fig.suptitle(
        "A/B Testing en Producción (Octubre 2026): Variante A (v2) vs. Variante B (v3) "
        "con Incertidumbre por Bootstrap de Bloques",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)
    return ruta_salida


if __name__ == "__main__":
    evaluar_ab_testing_octubre_2026()