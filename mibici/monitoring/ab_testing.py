"""
Módulo de entrenamiento de versión alternativa (v3), registro en MLflow Model Registry
y evaluación metodológica mediante A/B Testing (Variante A: v2 vs. Variante B: v3)
sobre el periodo actual de producción (Octubre 2026) para P1, P2 y P3 (Equipo 35).

Cumple los Pasos 4 y 5 de la actividad:
  4. Entrena una segunda versión del modelo (reentrenada con el histórico ampliado
     Octubre 2022 - Septiembre 2026, catálogo actualizado nomenclatura_2026_10.csv
     e hiperparámetros ajustados) y la registra en el Model Registry de MLflow como
     versión 3 (v3) de mibici-equipo35-p1-demanda, mibici-equipo35-p2-riesgo y
     mibici-equipo35-p3-grupos.
  5. Compara metodológicamente la versión actual (Variante A: v2) contra la nueva
     versión (Variante B: v3) sobre el mismo conjunto de evaluación de Octubre 2026,
     incluyendo pruebas de significancia estadística pareadas y gráficas comparativas.

Cómo ejecutarlo:
    python -m mibici.monitoring.ab_testing
"""

import json
from pathlib import Path
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from scipy import stats
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
        zip(df_nom_10["Estacion_Id"].astype("int64"), df_nom_10["dpcapacity"].astype("float64"))
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
    el Model Registry asignándoles el alias '@challenger' para el A/B Testing.
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    experimento = mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    cliente = MlflowClient()
    for run_f in cliente.search_runs([experimento.experiment_id], "attributes.status = 'FAILED'"):
        cliente.delete_run(run_f.info.run_id)

    git_commit, data_version_base = train.obtener_metadatos_trazabilidad()
    _, _, manifiesto_10 = drift_report.construir_features_octubre_2026()
    data_version_combinada = f"{data_version_base}+{manifiesto_10['data_version_short']}"

    # Revisamos si los 3 modelos ya cuentan con la versión 3 registrada
    versiones_p1 = [int(mv.version) for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P1}'")]
    versiones_p2 = [int(mv.version) for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P2}'")]
    versiones_p3 = [int(mv.version) for mv in cliente.search_model_versions(f"name='{config.MODEL_NAME_P3}'")]

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
        # Ponderamos con mayor peso las observaciones de 2026 para adaptar el nivel de demanda
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
                    "Reentrenada con 48 meses (Oct 2022 - Sep 2026) y nomenclatura_2026_10.csv."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P1, ver_p1.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P1, ver_p1.version, "data_version", data_version_combinada
            )
            print(f"  -> Registrado {config.MODEL_NAME_P1} v{ver_p1.version} (run_id={run_p1.info.run_id})")

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
                    "Reentrenada con 48 meses (Oct 2022 - Sep 2026) y nomenclatura_2026_10.csv."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P2, ver_p2.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P2, ver_p2.version, "data_version", data_version_combinada
            )
            print(f"  -> Registrado {config.MODEL_NAME_P2} v{ver_p2.version} (run_id={run_p2.info.run_id})")

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
                    "Reentrenada con perfiles 2026 (469 estaciones) y nomenclatura_2026_10.csv."
                ),
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P3, ver_p3.version, "git_commit", git_commit
            )
            cliente.set_model_version_tag(
                config.MODEL_NAME_P3, ver_p3.version, "data_version", data_version_combinada
            )
            print(f"  -> Registrado {config.MODEL_NAME_P3} v{ver_p3.version} (run_id={run_p3.info.run_id})")

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
# 4. EJECUTAR EL A/B TESTING SOBRE EL CONJUNTO DE EVALUACIÓN DE OCTUBRE 2026
# ==============================================================================
def evaluar_ab_testing_octubre_2026() -> dict:
    """
    Compara la Variante A (v2: modelo actual entrenado hasta Dic 2025) contra la
    Variante B (v3: modelo alternativo reentrenado hasta Sep 2026 con nomenclatura 2026_10)
    sobre todo el periodo de producción Octubre 2026 para P1, P2 y P3.
    """
    obtener_o_entrenar_version_v3()
    df_p1_p2_oct, df_p3_oct, manifiesto_10 = drift_report.construir_features_octubre_2026()

    cols_x_p1_p2 = drift_report.COLUMNAS_FEATURES_P1_P2
    cols_x_p3 = drift_report.COLUMNAS_FEATURES_P3

    X_eval_p1_p2 = df_p1_p2_oct[cols_x_p1_p2].astype("float64")
    y_eval_p1 = df_p1_p2_oct["viajes_iniciados"].astype("float64").to_numpy()
    y_eval_p2 = df_p1_p2_oct["en_riesgo"].astype("int64").to_numpy()
    mascara_pico = (df_p1_p2_oct["es_hora_pico"] == 1).to_numpy()

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P1 (REGRESIÓN DE DEMANDA HORARIA)
    # --------------------------------------------------------------------------
    m_p1_v2 = cargar_version_modelo_registry(config.MODEL_NAME_P1, config.AB_VARIANT_A_VERSION)
    m_p1_v3 = cargar_version_modelo_registry(config.MODEL_NAME_P1, config.AB_VARIANT_B_VERSION)

    pred_p1_v2 = np.clip(m_p1_v2.predict(X_eval_p1_p2), 0.0, None)
    pred_p1_v3 = np.clip(m_p1_v3.predict(X_eval_p1_p2), 0.0, None)

    err_abs_v2 = np.abs(y_eval_p1 - pred_p1_v2)
    err_abs_v3 = np.abs(y_eval_p1 - pred_p1_v3)

    # Prueba t pareada sobre los errores absolutos por observación (H1: MAE_v3 != MAE_v2)
    t_stat_p1, p_val_p1 = stats.ttest_rel(err_abs_v2, err_abs_v3)

    p1_v2_metrics = {
        "mae": round(float(mean_absolute_error(y_eval_p1, pred_p1_v2)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_eval_p1, pred_p1_v2))), 4),
        "r2": round(float(r2_score(y_eval_p1, pred_p1_v2)), 4),
        "peak_hour_mae": round(float(mean_absolute_error(y_eval_p1[mascara_pico], pred_p1_v2[mascara_pico])), 4),
        "mean_bias_error": round(float(np.mean(pred_p1_v2 - y_eval_p1)), 4),
        "pred_mean": round(float(np.mean(pred_p1_v2)), 4),
    }
    p1_v3_metrics = {
        "mae": round(float(mean_absolute_error(y_eval_p1, pred_p1_v3)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_eval_p1, pred_p1_v3))), 4),
        "r2": round(float(r2_score(y_eval_p1, pred_p1_v3)), 4),
        "peak_hour_mae": round(float(mean_absolute_error(y_eval_p1[mascara_pico], pred_p1_v3[mascara_pico])), 4),
        "mean_bias_error": round(float(np.mean(pred_p1_v3 - y_eval_p1)), 4),
        "pred_mean": round(float(np.mean(pred_p1_v3)), 4),
    }

    ganador_p1 = "v3" if p1_v3_metrics["mae"] < p1_v2_metrics["mae"] else "v2"

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P2 (CLASIFICACIÓN DE RIESGO DE DESBALANCE >= 30%)
    # --------------------------------------------------------------------------
    m_p2_v2 = cargar_version_modelo_registry(config.MODEL_NAME_P2, config.AB_VARIANT_A_VERSION)
    m_p2_v3 = cargar_version_modelo_registry(config.MODEL_NAME_P2, config.AB_VARIANT_B_VERSION)

    pred_p2_v2 = m_p2_v2.predict(X_eval_p1_p2)
    pred_p2_v3 = m_p2_v3.predict(X_eval_p1_p2)
    prob_p2_v2 = m_p2_v2.predict_proba(X_eval_p1_p2)[:, 1]
    prob_p2_v3 = m_p2_v3.predict_proba(X_eval_p1_p2)[:, 1]

    # Prueba t pareada sobre pérdida cuadrática de probabilidad (Brier loss por fila)
    brier_fila_v2 = (y_eval_p2 - prob_p2_v2) ** 2
    brier_fila_v3 = (y_eval_p2 - prob_p2_v3) ** 2
    t_stat_p2, p_val_p2 = stats.ttest_rel(brier_fila_v2, brier_fila_v3)

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

    ganador_p2 = (
        "v3"
        if (p2_v3_metrics["pr_auc"] >= p2_v2_metrics["pr_auc"] and p2_v3_metrics["f1"] >= p2_v2_metrics["f1"])
        else ("v3" if p2_v3_metrics["f1"] > p2_v2_metrics["f1"] else "v2")
    )

    # --------------------------------------------------------------------------
    # EVALUACIÓN A/B DE P3 (CLUSTERING DE ESTACIONES EN OCTUBRE 2026)
    # --------------------------------------------------------------------------
    m_p3_v2 = cargar_version_modelo_registry(config.MODEL_NAME_P3, config.AB_VARIANT_A_VERSION)
    m_p3_v3 = cargar_version_modelo_registry(config.MODEL_NAME_P3, config.AB_VARIANT_B_VERSION)

    X_eval_p3 = df_p3_oct[cols_x_p3].astype("float64")
    X_eval_p3_norm = StandardScaler().fit_transform(X_eval_p3)

    clusters_v2 = m_p3_v2.predict(X_eval_p3)
    clusters_v3 = m_p3_v3.predict(X_eval_p3)

    # Distancia euclidiana de cada estación a su centroide asignado en el espacio escalado de cada modelo
    X_trans_v2 = m_p3_v2.named_steps["scaler"].transform(X_eval_p3)
    centroides_v2 = m_p3_v2.named_steps["clusterer"].cluster_centers_
    dist_centroide_v2 = np.linalg.norm(X_trans_v2 - centroides_v2[clusters_v2], axis=1)

    X_trans_v3 = m_p3_v3.named_steps["scaler"].transform(X_eval_p3)
    centroides_v3 = m_p3_v3.named_steps["clusterer"].cluster_centers_
    dist_centroide_v3 = np.linalg.norm(X_trans_v3 - centroides_v3[clusters_v3], axis=1)

    t_stat_p3, p_val_p3 = stats.ttest_rel(dist_centroide_v2, dist_centroide_v3)

    prop_min_v2 = float(pd.Series(clusters_v2).value_counts().min() / len(X_eval_p3))
    prop_min_v3 = float(pd.Series(clusters_v3).value_counts().min() / len(X_eval_p3))

    p3_v2_metrics = {
        "silhouette_oct2026": round(float(silhouette_score(X_eval_p3_norm, clusters_v2)), 4),
        "davies_bouldin_oct2026": round(float(davies_bouldin_score(X_eval_p3_norm, clusters_v2)), 4),
        "mean_centroid_distance": round(float(np.mean(dist_centroide_v2)), 4),
        "min_cluster_share": round(prop_min_v2, 4),
        "cluster_counts": {str(k): int(v) for k, v in pd.Series(clusters_v2).value_counts().sort_index().items()},
    }
    p3_v3_metrics = {
        "silhouette_oct2026": round(float(silhouette_score(X_eval_p3_norm, clusters_v3)), 4),
        "davies_bouldin_oct2026": round(float(davies_bouldin_score(X_eval_p3_norm, clusters_v3)), 4),
        "mean_centroid_distance": round(float(np.mean(dist_centroide_v3)), 4),
        "min_cluster_share": round(prop_min_v3, 4),
        "cluster_counts": {str(k): int(v) for k, v in pd.Series(clusters_v3).value_counts().sort_index().items()},
    }

    ganador_p3 = (
        "v3"
        if p3_v3_metrics["silhouette_oct2026"] > p3_v2_metrics["silhouette_oct2026"]
        else "v2"
    )

    resultados_ab = {
        "evaluation_period": "Octubre 2026 (2026-10-01 a 2026-10-31)",
        "evaluation_data_version": manifiesto_10["data_version_short"],
        "hourly_observations_evaluated": int(len(X_eval_p1_p2)),
        "active_stations_evaluated_p3": int(len(X_eval_p3)),
        "models": {
            "p1": {
                "model_name": config.MODEL_NAME_P1,
                "real_target_mean_oct2026": round(float(np.mean(y_eval_p1)), 4),
                "variant_A_v2": p1_v2_metrics,
                "variant_B_v3": p1_v3_metrics,
                "paired_ttest_t_stat": round(float(t_stat_p1), 4),
                "paired_ttest_p_value": float(p_val_p1),
                "statistically_significant_p_lt_0_05": bool(p_val_p1 < 0.05),
                "selected_winner": ganador_p1,
            },
            "p2": {
                "model_name": config.MODEL_NAME_P2,
                "real_positive_rate_oct2026": round(float(np.mean(y_eval_p2)), 4),
                "variant_A_v2": p2_v2_metrics,
                "variant_B_v3": p2_v3_metrics,
                "paired_ttest_t_stat": round(float(t_stat_p2), 4),
                "paired_ttest_p_value": float(p_val_p2),
                "statistically_significant_p_lt_0_05": bool(p_val_p2 < 0.05),
                "selected_winner": ganador_p2,
            },
            "p3": {
                "model_name": config.MODEL_NAME_P3,
                "variant_A_v2": p3_v2_metrics,
                "variant_B_v3": p3_v3_metrics,
                "paired_ttest_t_stat": round(float(t_stat_p3), 4),
                "paired_ttest_p_value": float(p_val_p3),
                "statistically_significant_p_lt_0_05": bool(p_val_p3 < 0.05),
                "selected_winner": ganador_p3,
            },
        },
    }

    ruta_fig_ab = graficar_resultados_ab_testing(resultados_ab)
    resultados_ab["figure_path"] = str(ruta_fig_ab.relative_to(config.PROJ_ROOT))

    config.MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.AB_TEST_SUMMARY_JSON, "w", encoding="utf-8") as f_out:
        json.dump(resultados_ab, f_out, indent=4, ensure_ascii=False)

    print("\n" + "=" * 82)
    print("RESULTADOS DEL A/B TESTING SOBRE EL PERIODO ACTUAL (OCTUBRE 2026)")
    print("=" * 82)
    print(f"Observaciones horarias evaluadas (P1 y P2): {len(X_eval_p1_p2):,}")
    print(f"Estaciones activas evaluadas (P3):          {len(X_eval_p3):,}")

    print("\n1. P1 - Regresión de Demanda Horaria (mibici-equipo35-p1-demanda):")
    print(f"   * Variante A (v2 - Train hasta Dic 2025): {p1_v2_metrics}")
    print(f"   * Variante B (v3 - Reentrenada Sep 2026): {p1_v3_metrics}")
    print(f"   * Prueba t pareada: t={t_stat_p1:.4f}, p={p_val_p1:.4e} -> Ganador: {ganador_p1}")

    print("\n2. P2 - Clasificación de Riesgo Horario (mibici-equipo35-p2-riesgo):")
    print(f"   * Variante A (v2 - Train hasta Dic 2025): {p2_v2_metrics}")
    print(f"   * Variante B (v3 - Reentrenada Sep 2026): {p2_v3_metrics}")
    print(f"   * Prueba t pareada: t={t_stat_p2:.4f}, p={p_val_p2:.4e} -> Ganador: {ganador_p2}")

    print("\n3. P3 - Clustering de Perfiles de Estaciones (mibici-equipo35-p3-grupos):")
    print(f"   * Variante A (v2 - Train 373 est. 2025):  {p3_v2_metrics}")
    print(f"   * Variante B (v3 - Reentrenada 469 est.): {p3_v3_metrics}")
    print(f"   * Prueba t pareada: t={t_stat_p3:.4f}, p={p_val_p3:.4e} -> Ganador: {ganador_p3}")
    print("=" * 82)


    # Actualizamos los aliases (@champion y @challenger) en el Model Registry según el ganador
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
            cliente.set_registered_model_alias(nom_mod, config.PRODUCTION_ALIAS, ver_champ)
            cliente.set_registered_model_alias(nom_mod, config.CHALLENGER_ALIAS, ver_chall)

        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="AB_Testing_Evaluation_Oct2026"):
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "stage_of_work": "ab_testing",
                    "evaluation_period": "2026-10",
                    "winner_p1": resultados_ab["models"]["p1"]["selected_winner"],
                    "winner_p2": resultados_ab["models"]["p2"]["selected_winner"],
                    "winner_p3": resultados_ab["models"]["p3"]["selected_winner"],
                }
            )
            mlflow.log_metrics(
                {
                    "p1_v2_mae": resultados_ab["models"]["p1"]["variant_A_v2"]["mae"],
                    "p1_v3_mae": resultados_ab["models"]["p1"]["variant_B_v3"]["mae"],
                    "p2_v2_f1": resultados_ab["models"]["p2"]["variant_A_v2"]["f1"],
                    "p2_v3_f1": resultados_ab["models"]["p2"]["variant_B_v3"]["f1"],
                    "p2_v2_pr_auc": resultados_ab["models"]["p2"]["variant_A_v2"]["pr_auc"],
                    "p2_v3_pr_auc": resultados_ab["models"]["p2"]["variant_B_v3"]["pr_auc"],
                    "p3_v2_silhouette": resultados_ab["models"]["p3"]["variant_A_v2"]["silhouette_oct2026"],
                    "p3_v3_silhouette": resultados_ab["models"]["p3"]["variant_B_v3"]["silhouette_oct2026"],
                }
            )
            mlflow.log_artifact(str(config.AB_TEST_SUMMARY_JSON), artifact_path="monitoring")
            mlflow.log_artifact(str(ruta_fig_ab), artifact_path="plots")
    except Exception as exc:
        print(f"Aviso: No se pudieron actualizar aliases en MLflow ({exc}).")

    return resultados_ab


# ==============================================================================
# 5. GENERAR GRÁFICA COMPARATIVA DEL A/B TESTING PARA EL REPORTE Y EL README
# ==============================================================================
def graficar_resultados_ab_testing(resultados_ab: dict) -> Path:
    """
    Genera la gráfica comparativa de 3 paneles (P1, P2 y P3) contrastando las métricas
    de la Variante A (v2) vs. la Variante B (v3) sobre Octubre 2026.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta_salida = config.FIGURES_DIR / "ab_testing_comparacion_modelos.png"

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0), dpi=150)
    ancho = 0.34

    # Panel 1: P1 Regresión (MAE global, MAE Hora Pico, RMSE y Sesgo absoluto |Bias|)
    ax1 = axes[0]
    m1_a = resultados_ab["models"]["p1"]["variant_A_v2"]
    m1_b = resultados_ab["models"]["p1"]["variant_B_v3"]
    etiquetas_p1 = ["MAE\nGlobal", "MAE\nHora Pico", "RMSE", "|Sesgo|\nMedio"]
    vals_p1_a = [m1_a["mae"], m1_a["peak_hour_mae"], m1_a["rmse"], abs(m1_a["mean_bias_error"])]
    vals_p1_b = [m1_b["mae"], m1_b["peak_hour_mae"], m1_b["rmse"], abs(m1_b["mean_bias_error"])]
    x1 = np.arange(len(etiquetas_p1))
    ax1.bar(x1 - ancho / 2, vals_p1_a, width=ancho, label="Variante A (v2)", color="#2b8cbe", edgecolor="black")
    ax1.bar(x1 + ancho / 2, vals_p1_b, width=ancho, label="Variante B (v3)", color="#2ca25f", edgecolor="black")
    for idx, (va, vb) in enumerate(zip(vals_p1_a, vals_p1_b)):
        ax1.text(idx - ancho / 2, va + 0.05, f"{va:.2f}", ha="center", fontsize=7.8)
        ax1.text(idx + ancho / 2, vb + 0.05, f"{vb:.2f}", ha="center", fontsize=7.8, fontweight="bold")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(etiquetas_p1, fontsize=8.5)
    ax1.set_title("P1 Demanda (Menor = Mejor)", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, max(vals_p1_a + vals_p1_b) + 0.45)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: P2 Clasificación (F1, Precision, Recall, ROC-AUC, PR-AUC)
    ax2 = axes[1]
    m2_a = resultados_ab["models"]["p2"]["variant_A_v2"]
    m2_b = resultados_ab["models"]["p2"]["variant_B_v3"]
    etiquetas_p2 = ["F1-Score", "Precision", "Recall", "ROC-AUC", "PR-AUC"]
    vals_p2_a = [m2_a["f1"], m2_a["precision"], m2_a["recall"], m2_a["roc_auc"], m2_a["pr_auc"]]
    vals_p2_b = [m2_b["f1"], m2_b["precision"], m2_b["recall"], m2_b["roc_auc"], m2_b["pr_auc"]]
    x2 = np.arange(len(etiquetas_p2))
    ax2.bar(x2 - ancho / 2, vals_p2_a, width=ancho, label="Variante A (v2)", color="#2b8cbe", edgecolor="black")
    ax2.bar(x2 + ancho / 2, vals_p2_b, width=ancho, label="Variante B (v3)", color="#2ca25f", edgecolor="black")
    for idx, (va, vb) in enumerate(zip(vals_p2_a, vals_p2_b)):
        ax2.text(idx - ancho / 2 - 0.03, va + 0.018, f"{va:.3f}", ha="center", fontsize=6.8)
        ax2.text(idx + ancho / 2 + 0.03, vb + 0.038, f"{vb:.3f}", ha="center", fontsize=6.8, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(etiquetas_p2, fontsize=8.2)
    ax2.set_title("P2 Riesgo (Mayor = Mejor)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.08)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Panel 3: P3 Clustering (Silhouette Score, Distancia al centroide y Davies-Bouldin)
    ax3 = axes[2]
    m3_a = resultados_ab["models"]["p3"]["variant_A_v2"]
    m3_b = resultados_ab["models"]["p3"]["variant_B_v3"]
    etiquetas_p3 = ["Silhouette\n(Mayor mejor)", "Dist. Centroide\n(Menor mejor)", "Davies-Bouldin\n(Menor mejor)"]
    vals_p3_a = [m3_a["silhouette_oct2026"], m3_a["mean_centroid_distance"], m3_a["davies_bouldin_oct2026"]]
    vals_p3_b = [m3_b["silhouette_oct2026"], m3_b["mean_centroid_distance"], m3_b["davies_bouldin_oct2026"]]
    x3 = np.arange(len(etiquetas_p3))
    ax3.bar(x3 - ancho / 2, vals_p3_a, width=ancho, label="Variante A (v2)", color="#2b8cbe", edgecolor="black")
    ax3.bar(x3 + ancho / 2, vals_p3_b, width=ancho, label="Variante B (v3)", color="#2ca25f", edgecolor="black")
    for idx, (va, vb) in enumerate(zip(vals_p3_a, vals_p3_b)):
        ax3.text(idx - ancho / 2, va + 0.04, f"{va:.3f}", ha="center", fontsize=7.8)
        ax3.text(idx + ancho / 2, vb + 0.04, f"{vb:.3f}", ha="center", fontsize=7.8, fontweight="bold")
    ax3.set_xticks(x3)
    ax3.set_xticklabels(etiquetas_p3, fontsize=8.2)
    ax3.set_title("P3 Perfiles Estaciones (Oct 2026)", fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(vals_p3_a + vals_p3_b) + 0.35)
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Resultados de A/B Testing en Producción (Octubre 2026): Variante A (v2) vs. Variante B Reentrenada (v3)",
        fontsize=11.5,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(ruta_salida)
    plt.close(fig)
    return ruta_salida


if __name__ == "__main__":
    evaluar_ab_testing_octubre_2026()