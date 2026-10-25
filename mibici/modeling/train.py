"""
Script de entrenamiento parametrizado para el MLflow Project del Equipo 35.
Soporta la ejecución mediante 'mlflow run' para las tres preguntas (p1, p2 y p3)
en tres etapas de trabajo ('baseline', 'search' con nested runs, y 'candidate').

Registra en el servidor de MLflow Tracking:
  - Parámetros e hiperparámetros (con valores por defecto tomados de config.py).
  - Métricas de entrenamiento, validación, prueba y comparación contra baseline.
  - Etiquetas obligatorias: team, git_commit, question, model_family, stage_of_work.
  - Dataset de entrenamiento con mlflow.log_input (sin datos personales).
  - Artefactos organizados en subcarpetas:
      * data/   -> Manifiesto data_version.json
      * config/ -> Archivo config.py y resumen JSON de la corrida
      * plots/  -> Dos gráficas de desempeño por pregunta
      * model/  -> Modelo entrenado con firma (signature) y ejemplo de entrada
"""

import argparse
import json
import os
import subprocess
import time
import warnings

# Librerías de MLflow para tracking, datasets, modelos y firmas
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature

# IMPORTANTE: NO importamos LocalArtifactDatasetSource.
# MLflow crea esa clase por dentro mientras se ejecuta, así que no se puede importar.
# Basta con pasar la ruta del archivo como texto en "source=" (ver función 4).
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans, MiniBatchKMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
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

from mibici import config, features, plots


# ==============================================================================
# FUNCIÓN 1: OBTENER EL COMMIT ACTUAL DE GIT Y LA VERSIÓN SHA-256 DE LOS DATOS
# ==============================================================================
def obtener_metadatos_trazabilidad() -> tuple[str, str]:
    """
    Lee el hash corto del commit actual de Git y el código SHA-256 corto del
    archivo data/processed/data_version.json.
    """
    try:
        commit_git = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        commit_git = "sin_commit"

    version_datos = "desconocida"
    if config.DATA_VERSION_FILE.exists():
        with open(config.DATA_VERSION_FILE, "r", encoding="utf-8") as archivo_json:
            datos_version = json.load(archivo_json)
            version_datos = datos_version.get("data_version_short", "desconocida")

    return commit_git, version_datos


# ==============================================================================
# FUNCIÓN 2: CARGAR O CONSTRUIR LAS TABLAS DE FEATURES CON CACHÉ EN DATA/INTERIM
# ==============================================================================
def cargar_datos_p1_p2() -> pd.DataFrame:
    """
    Si la tabla de features para P1 y P2 ya existe en data/interim/, la carga
    directamente para ahorrar tiempo. Si no existe, la construye desde el Parquet
    limpio usando mibici/features.py y la guarda en data/interim/.
    """
    config.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.INTERIM_P1_P2_FILE.exists():
        print(f"Cargando tabla de features desde caché: {config.INTERIM_P1_P2_FILE.name}")
        return pd.read_parquet(config.INTERIM_P1_P2_FILE)

    print("Construyendo tabla horaria por estación desde datos_mibici.parquet...")
    columnas_necesarias = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    df_viajes = pd.read_parquet(config.PARQUET_FILE, columns=columnas_necesarias)
    tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes)
    df_p1_p2 = features.generar_features_p1_y_p2(tabla_horaria)
    df_p1_p2.to_parquet(config.INTERIM_P1_P2_FILE, index=False)
    print(f"Tabla guardada en caché: {config.INTERIM_P1_P2_FILE}")
    return df_p1_p2


def cargar_datos_p3() -> pd.DataFrame:
    """
    Si la tabla de perfil de estaciones para P3 ya existe en data/interim/, la carga.
    Si no existe, filtra únicamente los viajes de entrenamiento (hasta TRAIN_END_DATE)
    y construye la tabla usando mibici/features.py sin fuga de información.
    """
    config.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if config.INTERIM_P3_FILE.exists():
        print(f"Cargando tabla de estaciones P3 desde caché: {config.INTERIM_P3_FILE.name}")
        return pd.read_parquet(config.INTERIM_P3_FILE)

    print("Construyendo perfil de estaciones para P3 (solo periodo Train)...")
    columnas_necesarias = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    df_viajes = pd.read_parquet(config.PARQUET_FILE, columns=columnas_necesarias)
    fecha_corte_train = pd.Timestamp(config.TRAIN_END_DATE)
    viajes_train = df_viajes.loc[df_viajes["Inicio_del_viaje"] <= fecha_corte_train]
    df_p3 = features.generar_features_p3_clustering(viajes_train)
    df_p3.to_parquet(config.INTERIM_P3_FILE, index=False)
    print(f"Tabla P3 guardada en caché: {config.INTERIM_P3_FILE}")
    return df_p3


# ==============================================================================
# FUNCIÓN 3: LEER ARGUMENTOS DE LÍNEA DE COMANDOS CON DEFECTOS DE CONFIG.PY
# ==============================================================================
def leer_argumentos() -> argparse.Namespace:
    """
    Lee los parámetros enviados por 'mlflow run' y reemplaza cualquier valor
    'from_config' por el valor oficial definido en mibici/config.py.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento reproducible de modelos para P1, P2 y P3 (Equipo 35)."
    )
    parser.add_argument("--question", type=str, default="from_config")
    parser.add_argument("--stage", type=str, default="from_config")
    parser.add_argument("--model", type=str, default="from_config")
    parser.add_argument("--alpha", type=str, default="from_config")
    parser.add_argument("--c_reg", type=str, default="from_config")
    parser.add_argument("--n_estimators", type=str, default="from_config")
    parser.add_argument("--max_depth", type=str, default="from_config")
    parser.add_argument("--learning_rate", type=str, default="from_config")
    parser.add_argument("--n_clusters", type=str, default="from_config")

    args = parser.parse_args()

    # Reemplazamos "from_config" usando exclusivamente las constantes de config.py
    pregunta = config.DEFAULT_QUESTION if args.question == "from_config" else args.question.lower()
    etapa = config.DEFAULT_STAGE if args.stage == "from_config" else args.stage.lower()

    if args.model == "from_config":
        if etapa == "baseline":
            modelo = "dummy" if pregunta in ["p1", "p2"] else "kmeans"
        elif pregunta == "p1":
            modelo = config.DEFAULT_MODEL_P1
        elif pregunta == "p2":
            modelo = config.DEFAULT_MODEL_P2
        else:
            modelo = config.DEFAULT_MODEL_P3
    else:
        modelo = args.model.lower()

    alpha = config.DEFAULT_ALPHA if args.alpha == "from_config" else float(args.alpha)
    c_reg = config.DEFAULT_C_REG if args.c_reg == "from_config" else float(args.c_reg)
    n_estimators = (
        config.DEFAULT_N_ESTIMATORS
        if args.n_estimators == "from_config"
        else int(args.n_estimators)
    )
    max_depth = (
        config.DEFAULT_MAX_DEPTH
        if args.max_depth == "from_config"
        else int(args.max_depth)
    )
    learning_rate = (
        config.DEFAULT_LEARNING_RATE
        if args.learning_rate == "from_config"
        else float(args.learning_rate)
    )
    if args.n_clusters == "from_config":
        n_clusters = 2 if (pregunta == "p3" and etapa == "baseline") else config.DEFAULT_N_CLUSTERS
    else:
        n_clusters = int(args.n_clusters)

    return argparse.Namespace(
        question=pregunta,
        stage=etapa,
        model=modelo,
        alpha=alpha,
        c_reg=c_reg,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_clusters=n_clusters,
    )


# ==============================================================================
# FUNCIÓN 4: GUARDAR ARTEFACTOS COMUNES EN SUBCARPETAS (DATA/, CONFIG/, PLOTS/)
# ==============================================================================
def registrar_artefactos_y_modelo(
    modelo_entrenado,
    X_muestra: pd.DataFrame,
    pred_muestra: np.ndarray,
    rutas_graficas: list,
    resumen_config: dict,
    nombre_dataset: str,
    version_datos: str,
) -> None:
    """
    Guarda en MLflow todos los artefactos que piden los Pasos 5 y 6 de la tarea:
    dataset de entrenamiento, manifiesto de datos, configuración, gráficas y modelo.
    """

    # --------------------------------------------------------------------------
    # PASO 1. Registrar el conjunto de entrenamiento con mlflow.log_input
    # --------------------------------------------------------------------------
    # Tomamos solo 500 filas como muestra del conjunto de entrenamiento.
    # Estas columnas son variables agregadas por estación y hora (hora, mes,
    # capacidad, rezagos...). NO incluyen Usuario_Id, Genero ni Año_de_nacimiento,
    # así que no guardamos datos personales de usuarios de MiBici (Paso 6).
    df_input_seguro = X_muestra.head(500).copy().astype("float64")

    # Armamos la ruta del Parquet limpio que generó la ingesta (dataset.py).
    # La dejamos RELATIVA a la raíz del proyecto ("data/processed/datos_mibici.parquet")
    # para que sea la misma en la computadora de cada integrante del equipo.
    ruta_fuente = str(config.PARQUET_FILE.relative_to(config.PROJ_ROOT))

    # Creamos el dataset de MLflow:
    #   - source: la ruta como TEXTO. MLflow detecta que es un archivo local
    #             y crea internamente la fuente de tipo "local".
    #   - digest: el hash corto SHA-256 del manifiesto (data_version.json).
    #             Así cada corrida queda ligada a una versión exacta de los datos.
    # MLflow 3.17 registra dos veces la misma fuente de archivos locales
    # (una para rutas normales y otra para rutas "file://"). Por eso avisa que
    # la ruta "se puede interpretar de varias formas", aunque ambas opciones
    # son idénticas. Ocultamos SOLO ese aviso y SOLO en esta parte del código.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The specified dataset source can be interpreted in multiple ways",
            category=UserWarning,
        )
        # Creamos el dataset de MLflow con la ruta local y la versión de los datos
        dataset_mlflow = mlflow.data.from_pandas(
            df_input_seguro,
            source=ruta_fuente,
            name=nombre_dataset,
            digest=version_datos,
        )
    # Guardamos el dataset en la corrida con el contexto "training"
    mlflow.log_input(dataset_mlflow, context="training")

    # --------------------------------------------------------------------------
    # PASO 2. Subcarpeta "data/": manifiesto de versión de datos de la ingesta
    # --------------------------------------------------------------------------
    # Solo lo subimos si existe (lo crea "mlflow run . -e ingest")
    if config.DATA_VERSION_FILE.exists():
        mlflow.log_artifact(str(config.DATA_VERSION_FILE), artifact_path="data")

    # --------------------------------------------------------------------------
    # PASO 3. Subcarpeta "config/": archivos de configuración de la corrida
    # --------------------------------------------------------------------------
    # 3.1 Subimos el config.py completo (rutas, fechas de corte, semilla, etc.)
    ruta_config_py = config.PROJ_ROOT / "mibici" / "config.py"
    mlflow.log_artifact(str(ruta_config_py), artifact_path="config")

    # 3.2 Guardamos los parámetros exactos de esta corrida directamente en MLflow,
    #     sin crear un archivo local que se sobrescriba en cada corrida
    mlflow.log_dict(resumen_config, "config/run_config.json")

    # --------------------------------------------------------------------------
    # PASO 4. Subcarpeta "plots/": gráficas de desempeño de la pregunta
    # --------------------------------------------------------------------------
    # P1: real vs predicho y residuales | P2: matriz de confusión y curva P-R
    # P3: mapa y perfiles de los grupos de estaciones
    for ruta_fig in rutas_graficas:
        mlflow.log_artifact(str(ruta_fig), artifact_path="plots")

    # --------------------------------------------------------------------------
    # PASO 5. Modelo con log_model, firma (signature) y ejemplo de entrada
    # --------------------------------------------------------------------------
    # Ejemplo de entrada: 5 filas que muestran cómo deben llegar los datos al modelo
    ejemplo_entrada = X_muestra.head(5).copy().astype("float64")

    # La firma describe columnas y tipos de entrada, y el tipo de salida
    firma = infer_signature(
        X_muestra.head(50).copy().astype("float64"),
        pred_muestra[:50],
    )

    # Guardamos el modelo en MLflow con su firma y su ejemplo de entrada
    info_modelo = mlflow.sklearn.log_model(
        sk_model=modelo_entrenado,
        name="model",
        signature=firma,
        input_example=ejemplo_entrada,
        serialization_format="cloudpickle",
    )

    # Guardamos la URI del modelo como etiqueta.
    # predict.py la usa después para registrar esta versión en el Model Registry.
    mlflow.set_tag("logged_model_uri", info_modelo.model_uri)


# ==============================================================================
# FLUJO PRINCIPAL DE ENTRENAMIENTO (LINEAL PASO A PASO)
# ==============================================================================
def main() -> None:
    tiempo_inicio = time.time()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    args = leer_argumentos()
    git_commit, data_version = obtener_metadatos_trazabilidad()

    # Conectamos al servidor de MLflow configurado
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    # Si no venimos de 'mlflow run' (que ya define MLFLOW_RUN_ID), fijamos el experimento
    if not os.environ.get("MLFLOW_RUN_ID"):
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("=" * 72)
    print(
        f"INICIANDO CORRIDA -> Pregunta: {args.question.upper()} | "
        f"Etapa: {args.stage} | Modelo: {args.model}"
    )
    print("=" * 72)

    # ==========================================================================
    # CASO 1 Y 2: PREGUNTA 1 (REGRESIÓN) Y PREGUNTA 2 (CLASIFICACIÓN)
    # ==========================================================================
    if args.question in ["p1", "p2"]:
        df_p1_p2 = cargar_datos_p1_p2()

        # Partición estrictamente temporal usando las fechas de config.py
        fecha_corte_train = pd.Timestamp(config.TRAIN_END_DATE)
        fecha_corte_val = pd.Timestamp(config.VAL_END_DATE)

        mascara_train = df_p1_p2["Fecha_Hora"] <= fecha_corte_train
        mascara_val = (df_p1_p2["Fecha_Hora"] > fecha_corte_train) & (
            df_p1_p2["Fecha_Hora"] <= fecha_corte_val
        )
        mascara_test = df_p1_p2["Fecha_Hora"] > fecha_corte_val

        df_train = df_p1_p2.loc[mascara_train]
        df_val = df_p1_p2.loc[mascara_val]
        df_test = df_p1_p2.loc[mascara_test]

        # Variables predictoras agregadas por estación (sin datos personales de usuarios)
        columnas_x = [
            "hora",
            "dia_semana",
            "mes",
            "es_fin_de_semana",
            "es_hora_pico",
            "dpcapacity",
            "latitude",
            "longitude",
            "lag_1_iniciados",
            "lag_2_iniciados",
            "lag_24_iniciados",
            "media_movil_3_iniciados",
            "lag_1_terminados",
            "lag_1_flujo_neto",
            "lag_1_ratio_capacidad",
        ]

        X_train = df_train[columnas_x].astype("float64")
        X_val = df_val[columnas_x].astype("float64")
        X_test = df_test[columnas_x].astype("float64")

        # ----------------------------------------------------------------------
        # SUBCASO P1: REGRESIÓN DE DEMANDA HORARIA ('viajes_iniciados')
        # ----------------------------------------------------------------------
        if args.question == "p1":
            y_train = df_train["viajes_iniciados"].astype("float64")
            y_val = df_val["viajes_iniciados"].astype("float64")
            y_test = df_test["viajes_iniciados"].astype("float64")

            # Calculamos la referencia del Baseline (DummyRegressor) para comparar
            base_reg = DummyRegressor(strategy="mean")
            base_reg.fit(X_train, y_train)
            base_val_mae = float(mean_absolute_error(y_val, base_reg.predict(X_val)))
            base_test_mae = float(mean_absolute_error(y_test, base_reg.predict(X_test)))
            base_test_rmse = float(np.sqrt(mean_squared_error(y_test, base_reg.predict(X_test))))

            # A) BÚSQUEDA DE HIPERPARÁMETROS CON NESTED RUNS PARA P1
            if args.stage == "search":
                nombre_padre = "P1_Search_HistGradientBoosting"
                with mlflow.start_run(run_name=nombre_padre) as run_padre:
                    mlflow.set_tag("mlflow.runName", nombre_padre)
                    mlflow.set_tags(
                        {
                            "team": config.TEAM_TAG,
                            "git_commit": git_commit,
                            "data_version": data_version,
                            "question": "p1",
                            "model_family": "GradientBoosting",
                            "stage_of_work": "search",
                        }
                    )
                    mlflow.log_params(
                        {
                            "question": "p1",
                            "search_strategy": "grid_nested_runs",
                            "model_type": "HistGradientBoostingRegressor",
                            "num_combinations": len(config.SEARCH_GRID_P1),
                            "date_range": config.DATE_RANGE_LABEL,
                            "data_version": data_version,
                            "random_seed": config.RANDOM_SEED,
                        }
                    )

                    mejor_val_mae = 999999.0
                    mejor_comb = {}

                    for idx, comb in enumerate(config.SEARCH_GRID_P1, start=1):
                        nombre_hija = (
                            f"P1_Trial{idx}_HGB_iter{comb['n_estimators']}_depth{comb['max_depth']}"
                        )
                        with mlflow.start_run(run_name=nombre_hija, nested=True) as run_hija:
                            mlflow.set_tag("mlflow.runName", nombre_hija)
                            mlflow.set_tags(
                                {
                                    "team": config.TEAM_TAG,
                                    "git_commit": git_commit,
                                    "data_version": data_version,
                                    "question": "p1",
                                    "model_family": "GradientBoosting",
                                    "stage_of_work": "search",
                                }
                            )
                            modelo_hijo = HistGradientBoostingRegressor(
                                max_iter=comb["n_estimators"],
                                max_depth=comb["max_depth"],
                                learning_rate=comb["learning_rate"],
                                random_state=config.RANDOM_SEED,
                            )
                            modelo_hijo.fit(X_train, y_train)
                            pred_v = np.clip(modelo_hijo.predict(X_val), 0, None)
                            pred_t = np.clip(modelo_hijo.predict(X_test), 0, None)

                            v_mae = float(mean_absolute_error(y_val, pred_v))
                            v_rmse = float(np.sqrt(mean_squared_error(y_val, pred_v)))
                            t_mae = float(mean_absolute_error(y_test, pred_t))
                            t_rmse = float(np.sqrt(mean_squared_error(y_test, pred_t)))
                            t_r2 = float(r2_score(y_test, pred_t))

                            mlflow.log_params(
                                {
                                    "model_type": "HistGradientBoostingRegressor",
                                    "n_estimators": comb["n_estimators"],
                                    "max_depth": comb["max_depth"],
                                    "learning_rate": comb["learning_rate"],
                                    "random_seed": config.RANDOM_SEED,
                                }
                            )
                            mlflow.log_metrics(
                                {
                                    "val_mae": v_mae,
                                    "val_rmse": v_rmse,
                                    "test_mae": t_mae,
                                    "test_rmse": t_rmse,
                                    "test_r2": t_r2,
                                }
                            )
                            print(
                                f"  [Hija {idx}] {nombre_hija} (run_id={run_hija.info.run_id}) "
                                f"-> val_mae={v_mae:.4f} | test_mae={t_mae:.4f}"
                            )
                            if v_mae < mejor_val_mae:
                                mejor_val_mae = v_mae
                                mejor_comb = comb

                    mlflow.log_metrics(
                        {
                            "best_child_val_mae": mejor_val_mae,
                            "baseline_val_mae": base_val_mae,
                        }
                    )
                    mlflow.log_params(
                        {
                            "best_n_estimators": mejor_comb["n_estimators"],
                            "best_max_depth": mejor_comb["max_depth"],
                            "best_learning_rate": mejor_comb["learning_rate"],
                        }
                    )
                    print(
                        f"Búsqueda P1 completada (Padre run_id={run_padre.info.run_id}). "
                        f"Mejor combinación: {mejor_comb} con val_mae={mejor_val_mae:.4f}"
                    )
                return

            # B) ENTRENAMIENTO DE BASELINE O CORRIDA CANDIDATA PARA P1
            if args.stage == "baseline" or args.model == "dummy":
                modelo_p1 = DummyRegressor(strategy="mean")
                familia_modelo = "Dummy"
                etiqueta_etapa = "baseline"
                nombre_corrida = "P1_Baseline_DummyRegressor"
                hiperparametros = {"strategy": "mean"}
            elif args.model == "ridge":
                modelo_p1 = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("reg", Ridge(alpha=args.alpha, random_state=config.RANDOM_SEED)),
                    ]
                )
                familia_modelo = "Linear"
                etiqueta_etapa = "candidate"
                nombre_corrida = f"P1_Candidato_Ridge_alpha{args.alpha}"
                hiperparametros = {"alpha": args.alpha}
            elif args.model == "lasso":
                modelo_p1 = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("reg", Lasso(alpha=args.alpha, random_state=config.RANDOM_SEED)),
                    ]
                )
                familia_modelo = "Linear"
                etiqueta_etapa = "candidate"
                nombre_corrida = f"P1_Candidato_Lasso_alpha{args.alpha}"
                hiperparametros = {"alpha": args.alpha}
            elif args.model == "rf":
                modelo_p1 = RandomForestRegressor(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    max_samples=config.RF_MAX_SAMPLES,
                    n_jobs=-1,
                    random_state=config.RANDOM_SEED,
                )
                familia_modelo = "RandomForest"
                etiqueta_etapa = "candidate"
                nombre_corrida = (
                    f"P1_Candidato_RandomForest_n{args.n_estimators}_d{args.max_depth}"
                )
                hiperparametros = {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "max_samples": config.RF_MAX_SAMPLES,
                }
            elif args.model == "hgb":
                modelo_p1 = HistGradientBoostingRegressor(
                    max_iter=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    random_state=config.RANDOM_SEED,
                )
                familia_modelo = "GradientBoosting"
                etiqueta_etapa = "candidate"
                nombre_corrida = (
                    f"P1_Candidato_HistGB_iter{args.n_estimators}_d{args.max_depth}_lr{args.learning_rate}"
                )
                hiperparametros = {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                }
            else:
                raise ValueError(f"Modelo no reconocido para P1: {args.model}")

            modelo_p1.fit(X_train, y_train)
            pred_train = np.clip(modelo_p1.predict(X_train), 0, None)
            pred_val = np.clip(modelo_p1.predict(X_val), 0, None)
            pred_test = np.clip(modelo_p1.predict(X_test), 0, None)

            val_mae = float(mean_absolute_error(y_val, pred_val))
            test_mae = float(mean_absolute_error(y_test, pred_test))
            mejora_test_mae_pct = float(((base_test_mae - test_mae) / base_test_mae) * 100.0)

            metricas = {
                "baseline_val_mae": base_val_mae,
                "baseline_test_mae": base_test_mae,
                "baseline_test_rmse": base_test_rmse,
                "train_mae": float(mean_absolute_error(y_train, pred_train)),
                "val_mae": val_mae,
                "val_rmse": float(np.sqrt(mean_squared_error(y_val, pred_val))),
                "val_r2": float(r2_score(y_val, pred_val)),
                "test_mae": test_mae,
                "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
                "test_r2": float(r2_score(y_test, pred_test)),
                "improvement_over_baseline_pct": mejora_test_mae_pct,
                "training_time_sec": round(time.time() - tiempo_inicio, 2),
            }

            ruta_fig1 = plots.graficar_real_vs_predicho_p1(
                y_real=y_test,
                y_pred=pred_test,
                ruta_salida=config.FIGURES_DIR / "p1_regresion_real_vs_predicho.png",
                semilla=config.RANDOM_SEED,
                nombre_modelo=args.model.upper(),
            )
            ruta_fig2 = plots.graficar_residuales_p1(
                y_real=y_test,
                y_pred=pred_test,
                ruta_salida=config.FIGURES_DIR / "p1_regresion_residuales.png",
                semilla=config.RANDOM_SEED,
                nombre_modelo=args.model.upper(),
            )

            with mlflow.start_run(run_name=nombre_corrida) as run_actual:
                mlflow.set_tag("mlflow.runName", nombre_corrida)
                mlflow.set_tags(
                    {
                        "team": config.TEAM_TAG,
                        "git_commit": git_commit,
                        "data_version": data_version,
                        "question": "p1",
                        "model_family": familia_modelo,
                        "stage_of_work": etiqueta_etapa,
                    }
                )
                parametros_log = {
                    "question": "p1",
                    "model_name": args.model,
                    "date_range": config.DATE_RANGE_LABEL,
                    "data_version": data_version,
                    "random_seed": config.RANDOM_SEED,
                    "train_end_date": config.TRAIN_END_DATE,
                    "val_end_date": config.VAL_END_DATE,
                    **hiperparametros,
                }
                mlflow.log_params(parametros_log)
                mlflow.log_metrics(metricas)

                registrar_artefactos_y_modelo(
                    modelo_entrenado=modelo_p1,
                    X_muestra=X_train,
                    pred_muestra=pred_train,
                    rutas_graficas=[ruta_fig1, ruta_fig2],
                    resumen_config=parametros_log,
                    nombre_dataset="mibici_p1_estacion_hora_train",
                    version_datos=data_version,
                )

                print(f"Corrida registrada: {nombre_corrida}")
                print(f"  run_id: {run_actual.info.run_id}")
                for k, v in metricas.items():
                    print(f"  {k}: {v:.4f}")

        # ----------------------------------------------------------------------
        # SUBCASO P2: CLASIFICACIÓN DE RIESGO HORARIO ('en_riesgo')
        # ----------------------------------------------------------------------
        elif args.question == "p2":
            y_train = df_train["en_riesgo"].astype("int64")
            y_val = df_val["en_riesgo"].astype("int64")
            y_test = df_test["en_riesgo"].astype("int64")

            # Calculamos la referencia del Baseline (DummyClassifier estratificado)
            base_clf = DummyClassifier(
                strategy="stratified", random_state=config.RANDOM_SEED
            )
            base_clf.fit(X_train, y_train)
            base_val_f1 = float(f1_score(y_val, base_clf.predict(X_val), zero_division=0))
            base_test_f1 = float(f1_score(y_test, base_clf.predict(X_test), zero_division=0))
            base_test_recall = float(
                recall_score(y_test, base_clf.predict(X_test), zero_division=0)
            )

            # A) BÚSQUEDA DE HIPERPARÁMETROS CON NESTED RUNS PARA P2
            if args.stage == "search":
                nombre_padre = "P2_Search_HistGradientBoosting_Clf"
                with mlflow.start_run(run_name=nombre_padre) as run_padre:
                    mlflow.set_tag("mlflow.runName", nombre_padre)
                    mlflow.set_tags(
                        {
                            "team": config.TEAM_TAG,
                            "git_commit": git_commit,
                            "data_version": data_version,
                            "question": "p2",
                            "model_family": "GradientBoosting",
                            "stage_of_work": "search",
                        }
                    )
                    mlflow.log_params(
                        {
                            "question": "p2",
                            "search_strategy": "grid_nested_runs",
                            "model_type": "HistGradientBoostingClassifier",
                            "num_combinations": len(config.SEARCH_GRID_P2),
                            "date_range": config.DATE_RANGE_LABEL,
                            "data_version": data_version,
                            "random_seed": config.RANDOM_SEED,
                        }
                    )

                    mejor_val_f1 = -1.0
                    mejor_comb = {}

                    for idx, comb in enumerate(config.SEARCH_GRID_P2, start=1):
                        nombre_hija = (
                            f"P2_Trial{idx}_HGB_iter{comb['n_estimators']}_depth{comb['max_depth']}"
                        )
                        with mlflow.start_run(run_name=nombre_hija, nested=True) as run_hija:
                            mlflow.set_tag("mlflow.runName", nombre_hija)
                            mlflow.set_tags(
                                {
                                    "team": config.TEAM_TAG,
                                    "git_commit": git_commit,
                                    "data_version": data_version,
                                    "question": "p2",
                                    "model_family": "GradientBoosting",
                                    "stage_of_work": "search",
                                }
                            )
                            modelo_hijo = HistGradientBoostingClassifier(
                                class_weight="balanced",
                                max_iter=comb["n_estimators"],
                                max_depth=comb["max_depth"],
                                learning_rate=comb["learning_rate"],
                                random_state=config.RANDOM_SEED,
                            )
                            modelo_hijo.fit(X_train, y_train)
                            pred_v = modelo_hijo.predict(X_val)
                            pred_t = modelo_hijo.predict(X_test)
                            prob_v = modelo_hijo.predict_proba(X_val)[:, 1]
                            prob_t = modelo_hijo.predict_proba(X_test)[:, 1]

                            v_f1 = float(f1_score(y_val, pred_v, zero_division=0))
                            v_rec = float(recall_score(y_val, pred_v, zero_division=0))
                            v_auc = float(roc_auc_score(y_val, prob_v))
                            t_f1 = float(f1_score(y_test, pred_t, zero_division=0))
                            t_rec = float(recall_score(y_test, pred_t, zero_division=0))
                            t_auc = float(roc_auc_score(y_test, prob_t))

                            mlflow.log_params(
                                {
                                    "model_type": "HistGradientBoostingClassifier",
                                    "n_estimators": comb["n_estimators"],
                                    "max_depth": comb["max_depth"],
                                    "learning_rate": comb["learning_rate"],
                                    "class_weight": "balanced",
                                }
                            )
                            mlflow.log_metrics(
                                {
                                    "val_f1": v_f1,
                                    "val_recall": v_rec,
                                    "val_roc_auc": v_auc,
                                    "test_f1": t_f1,
                                    "test_recall": t_rec,
                                    "test_roc_auc": t_auc,
                                }
                            )
                            print(
                                f"  [Hija {idx}] {nombre_hija} (run_id={run_hija.info.run_id}) "
                                f"-> val_f1={v_f1:.4f} | test_f1={t_f1:.4f} | test_auc={t_auc:.4f}"
                            )
                            if v_f1 > mejor_val_f1:
                                mejor_val_f1 = v_f1
                                mejor_comb = comb

                    mlflow.log_metrics(
                        {
                            "best_child_val_f1": mejor_val_f1,
                            "baseline_val_f1": base_val_f1,
                        }
                    )
                    mlflow.log_params(
                        {
                            "best_n_estimators": mejor_comb["n_estimators"],
                            "best_max_depth": mejor_comb["max_depth"],
                            "best_learning_rate": mejor_comb["learning_rate"],
                        }
                    )
                    print(
                        f"Búsqueda P2 completada (Padre run_id={run_padre.info.run_id}). "
                        f"Mejor combinación: {mejor_comb} con val_f1={mejor_val_f1:.4f}"
                    )
                return

            # B) ENTRENAMIENTO DE BASELINE O CORRIDA CANDIDATA PARA P2
            if args.stage == "baseline" or args.model == "dummy":
                modelo_p2 = DummyClassifier(
                    strategy="stratified", random_state=config.RANDOM_SEED
                )
                familia_modelo = "Dummy"
                etiqueta_etapa = "baseline"
                nombre_corrida = "P2_Baseline_DummyClassifier"
                hiperparametros = {"strategy": "stratified"}
            elif args.model == "logreg":
                modelo_p2 = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=args.c_reg,
                                class_weight="balanced",
                                max_iter=200,
                                random_state=config.RANDOM_SEED,
                            ),
                        ),
                    ]
                )
                familia_modelo = "Linear"
                etiqueta_etapa = "candidate"
                nombre_corrida = f"P2_Candidato_LogReg_C{args.c_reg}"
                hiperparametros = {"C": args.c_reg, "class_weight": "balanced"}
            elif args.model == "rf_clf":
                modelo_p2 = RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    max_samples=config.RF_MAX_SAMPLES,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=config.RANDOM_SEED,
                )
                familia_modelo = "RandomForest"
                etiqueta_etapa = "candidate"
                nombre_corrida = (
                    f"P2_Candidato_RandomForest_n{args.n_estimators}_d{args.max_depth}"
                )
                hiperparametros = {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "max_samples": config.RF_MAX_SAMPLES,
                    "class_weight": "balanced",
                }
            elif args.model == "hgb_clf":
                modelo_p2 = HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=args.n_estimators,
                    max_depth=args.max_depth,
                    learning_rate=args.learning_rate,
                    random_state=config.RANDOM_SEED,
                )
                familia_modelo = "GradientBoosting"
                etiqueta_etapa = "candidate"
                nombre_corrida = (
                    f"P2_Candidato_HistGB_iter{args.n_estimators}_d{args.max_depth}_lr{args.learning_rate}"
                )
                hiperparametros = {
                    "n_estimators": args.n_estimators,
                    "max_depth": args.max_depth,
                    "learning_rate": args.learning_rate,
                    "class_weight": "balanced",
                }
            else:
                raise ValueError(f"Modelo no reconocido para P2: {args.model}")

            modelo_p2.fit(X_train, y_train)
            pred_train = modelo_p2.predict(X_train)
            pred_val = modelo_p2.predict(X_val)
            pred_test = modelo_p2.predict(X_test)
            prob_val = modelo_p2.predict_proba(X_val)[:, 1]
            prob_test = modelo_p2.predict_proba(X_test)[:, 1]

            test_f1 = float(f1_score(y_test, pred_test, zero_division=0))
            mejora_f1_veces = float(test_f1 / max(base_test_f1, 1e-6))

            metricas = {
                "positive_class_ratio_train": float(y_train.mean()),
                "baseline_val_f1": base_val_f1,
                "baseline_test_f1": base_test_f1,
                "baseline_test_recall": base_test_recall,
                "val_f1": float(f1_score(y_val, pred_val, zero_division=0)),
                "val_precision": float(precision_score(y_val, pred_val, zero_division=0)),
                "val_recall": float(recall_score(y_val, pred_val, zero_division=0)),
                "val_roc_auc": float(roc_auc_score(y_val, prob_val)),
                "val_pr_auc": float(average_precision_score(y_val, prob_val)),
                "test_f1": test_f1,
                "test_precision": float(precision_score(y_test, pred_test, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred_test, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, prob_test)),
                "test_pr_auc": float(average_precision_score(y_test, prob_test)),
                "improvement_f1_ratio_vs_baseline": mejora_f1_veces,
                "training_time_sec": round(time.time() - tiempo_inicio, 2),
            }

            ruta_fig1 = plots.graficar_matriz_confusion_p2(
                y_real=y_test,
                y_pred=pred_test,
                ruta_salida=config.FIGURES_DIR / "p2_clasificacion_matriz_confusion.png",
                nombre_modelo=args.model.upper(),
            )
            ruta_fig2 = plots.graficar_curva_precision_recall_p2(
                y_real=y_test,
                y_prob=prob_test,
                ruta_salida=config.FIGURES_DIR / "p2_clasificacion_precision_recall.png",
                nombre_modelo=args.model.upper(),
            )

            with mlflow.start_run(run_name=nombre_corrida) as run_actual:
                mlflow.set_tag("mlflow.runName", nombre_corrida)
                mlflow.set_tags(
                    {
                        "team": config.TEAM_TAG,
                        "git_commit": git_commit,
                        "data_version": data_version,
                        "question": "p2",
                        "model_family": familia_modelo,
                        "stage_of_work": etiqueta_etapa,
                    }
                )
                parametros_log = {
                    "question": "p2",
                    "model_name": args.model,
                    "risk_capacity_threshold": config.RISK_CAPACITY_RATIO_THRESHOLD,
                    "date_range": config.DATE_RANGE_LABEL,
                    "data_version": data_version,
                    "random_seed": config.RANDOM_SEED,
                    **hiperparametros,
                }
                mlflow.log_params(parametros_log)
                mlflow.log_metrics(metricas)

                registrar_artefactos_y_modelo(
                    modelo_entrenado=modelo_p2,
                    X_muestra=X_train,
                    pred_muestra=pred_train,
                    rutas_graficas=[ruta_fig1, ruta_fig2],
                    resumen_config=parametros_log,
                    nombre_dataset="mibici_p2_estacion_hora_train",
                    version_datos=data_version,
                )

                print(f"Corrida registrada: {nombre_corrida}")
                print(f"  run_id: {run_actual.info.run_id}")
                for k, v in metricas.items():
                    print(f"  {k}: {v:.4f}")

    # ==========================================================================
    # CASO 3: PREGUNTA 3 (CLUSTERING DE ESTACIONES)
    # ==========================================================================
    elif args.question == "p3":
        df_p3 = cargar_datos_p3()

        columnas_x_p3 = [
            "promedio_salidas_diarias",
            "duracion_mediana_min",
            "prop_fin_de_semana",
            "prop_pico_manana",
            "prop_pico_tarde",
            "ratio_salidas_llegadas",
            "dpcapacity",
            "latitude",
            "longitude",
        ]

        X_p3 = df_p3[columnas_x_p3].astype("float64")
        escalador_p3 = StandardScaler()
        X_p3_scaled = escalador_p3.fit_transform(X_p3)

        # Calculamos el Baseline de referencia (KMeans con K=2)
        kmeans_base = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
        etiquetas_k2 = kmeans_base.fit_predict(X_p3_scaled)
        base_sil = float(silhouette_score(X_p3_scaled, etiquetas_k2))
        base_db = float(davies_bouldin_score(X_p3_scaled, etiquetas_k2))

        # A) BÚSQUEDA DE HIPERPARÁMETROS CON NESTED RUNS PARA P3
        if args.stage == "search":
            nombre_padre = "P3_Search_KMeans_Clusters"
            with mlflow.start_run(run_name=nombre_padre) as run_padre:
                mlflow.set_tag("mlflow.runName", nombre_padre)
                mlflow.set_tags(
                    {
                        "team": config.TEAM_TAG,
                        "git_commit": git_commit,
                        "data_version": data_version,
                        "question": "p3",
                        "model_family": "KMeans",
                        "stage_of_work": "search",
                    }
                )
                mlflow.log_params(
                    {
                        "question": "p3",
                        "search_strategy": "grid_nested_runs",
                        "model_type": "KMeans",
                        "num_combinations": len(config.SEARCH_GRID_P3),
                        "date_range": config.DATE_RANGE_LABEL,
                        "data_version": data_version,
                        "random_seed": config.RANDOM_SEED,
                    }
                )

                mejor_sil = -1.0
                mejor_k = 4

                for idx, comb in enumerate(config.SEARCH_GRID_P3, start=1):
                    k_val = comb["n_clusters"]
                    nombre_hija = f"P3_Trial{idx}_KMeans_K{k_val}"
                    with mlflow.start_run(run_name=nombre_hija, nested=True) as run_hija:
                        mlflow.set_tag("mlflow.runName", nombre_hija)
                        mlflow.set_tags(
                            {
                                "team": config.TEAM_TAG,
                                "git_commit": git_commit,
                                "data_version": data_version,
                                "question": "p3",
                                "model_family": "KMeans",
                                "stage_of_work": "search",
                            }
                        )
                        km_hijo = KMeans(
                            n_clusters=k_val, n_init=10, random_state=config.RANDOM_SEED
                        )
                        etiq_hijo = km_hijo.fit_predict(X_p3_scaled)
                        sil_h = float(silhouette_score(X_p3_scaled, etiq_hijo))
                        db_h = float(davies_bouldin_score(X_p3_scaled, etiq_hijo))
                        inercia_h = float(km_hijo.inertia_)

                        mlflow.log_params(
                            {
                                "model_type": "KMeans",
                                "n_clusters": k_val,
                                "n_init": 10,
                                "random_seed": config.RANDOM_SEED,
                            }
                        )
                        mlflow.log_metrics(
                            {
                                "train_silhouette": sil_h,
                                "train_davies_bouldin": db_h,
                                "train_inertia": inercia_h,
                            }
                        )
                        print(
                            f"  [Hija {idx}] {nombre_hija} (run_id={run_hija.info.run_id}) "
                            f"-> silhouette={sil_h:.4f} | davies_bouldin={db_h:.4f}"
                        )
                        if sil_h > mejor_sil:
                            mejor_sil = sil_h
                            mejor_k = k_val

                mlflow.log_metrics(
                    {
                        "best_child_silhouette": mejor_sil,
                        "baseline_k2_train_silhouette": base_sil,
                    }
                )
                mlflow.log_params({"best_n_clusters": mejor_k})
                print(
                    f"Búsqueda P3 completada (Padre run_id={run_padre.info.run_id}). "
                    f"Mejor K en candidatos: K={mejor_k} con silhouette={mejor_sil:.4f}"
                )
            return

        # B) ENTRENAMIENTO DE BASELINE O CORRIDA CANDIDATA PARA P3
        if args.stage == "baseline":
            estimador_cluster = KMeans(
                n_clusters=2, n_init=10, random_state=config.RANDOM_SEED
            )
            familia_modelo = "KMeans"
            etiqueta_etapa = "baseline"
            nombre_corrida = "P3_Baseline_KMeans_K2"
            hiperparametros = {"n_clusters": 2, "n_init": 10}
        elif args.model == "kmeans":
            estimador_cluster = KMeans(
                n_clusters=args.n_clusters, n_init=10, random_state=config.RANDOM_SEED
            )
            familia_modelo = "KMeans"
            etiqueta_etapa = "candidate"
            nombre_corrida = f"P3_Candidato_KMeans_K{args.n_clusters}"
            hiperparametros = {"n_clusters": args.n_clusters, "n_init": 10}
        elif args.model == "minibatch_kmeans":
            estimador_cluster = MiniBatchKMeans(
                n_clusters=args.n_clusters,
                batch_size=128,
                n_init=10,
                random_state=config.RANDOM_SEED,
            )
            familia_modelo = "MiniBatchKMeans"
            etiqueta_etapa = "candidate"
            nombre_corrida = f"P3_Candidato_MiniBatchKMeans_K{args.n_clusters}"
            hiperparametros = {"n_clusters": args.n_clusters, "batch_size": 128}
        elif args.model == "birch":
            estimador_cluster = Birch(n_clusters=args.n_clusters, threshold=0.5)
            familia_modelo = "Hierarchical_Birch"
            etiqueta_etapa = "candidate"
            nombre_corrida = f"P3_Candidato_Birch_K{args.n_clusters}"
            hiperparametros = {"n_clusters": args.n_clusters, "threshold": 0.5}
        else:
            raise ValueError(f"Modelo no reconocido para P3: {args.model}")

        modelo_p3 = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clusterer", estimador_cluster),
            ]
        )
        modelo_p3.fit(X_p3)
        etiquetas_pred = modelo_p3.predict(X_p3)
        df_p3_resultado = df_p3.copy()
        df_p3_resultado["cluster"] = etiquetas_pred

        sil_actual = float(silhouette_score(X_p3_scaled, etiquetas_pred))
        db_actual = float(davies_bouldin_score(X_p3_scaled, etiquetas_pred))
        inercia_actual = float(
            getattr(estimador_cluster, "inertia_", 0.0)
        )

        # Medimos el balance del tamaño de los grupos (proporción del grupo más pequeño)
        conteos_grupos = pd.Series(etiquetas_pred).value_counts()
        prop_grupo_minimo = float(conteos_grupos.min() / len(df_p3))

        metricas = {
            "baseline_k2_train_silhouette": base_sil,
            "baseline_k2_train_davies_bouldin": base_db,
            "train_silhouette": sil_actual,
            "train_davies_bouldin": db_actual,
            "train_inertia": inercia_actual,
            "min_cluster_share": prop_grupo_minimo,
            "stations_clustered": float(len(df_p3)),
            "training_time_sec": round(time.time() - tiempo_inicio, 2),
        }

        ruta_fig1 = plots.graficar_mapa_clusters_p3(
            df_p3=df_p3_resultado,
            ruta_salida=config.FIGURES_DIR / "p3_clustering_mapa_estaciones.png",
            nombre_modelo=f"{args.model.upper()} K={hiperparametros['n_clusters']}",
        )
        ruta_fig2 = plots.graficar_perfiles_clusters_p3(
            df_p3=df_p3_resultado,
            columnas_perfil=columnas_x_p3,
            ruta_salida=config.FIGURES_DIR / "p3_clustering_perfiles_estaciones.png",
            nombre_modelo=f"{args.model.upper()} K={hiperparametros['n_clusters']}",
        )

        with mlflow.start_run(run_name=nombre_corrida) as run_actual:
            mlflow.set_tag("mlflow.runName", nombre_corrida)
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "git_commit": git_commit,
                    "data_version": data_version,
                    "question": "p3",
                    "model_family": familia_modelo,
                    "stage_of_work": etiqueta_etapa,
                }
            )
            parametros_log = {
                "question": "p3",
                "model_name": args.model,
                "date_range": config.DATE_RANGE_LABEL,
                "data_version": data_version,
                "random_seed": config.RANDOM_SEED,
                **hiperparametros,
            }
            mlflow.log_params(parametros_log)
            mlflow.log_metrics(metricas)

            registrar_artefactos_y_modelo(
                modelo_entrenado=modelo_p3,
                X_muestra=X_p3,
                pred_muestra=etiquetas_pred,
                rutas_graficas=[ruta_fig1, ruta_fig2],
                resumen_config=parametros_log,
                nombre_dataset="mibici_p3_perfil_estaciones_train",
                version_datos=data_version,
            )

            print(f"Corrida registrada: {nombre_corrida}")
            print(f"  run_id: {run_actual.info.run_id}")
            for k, v in metricas.items():
                print(f"  {k}: {v:.4f}")
            print(f"  Distribución por grupo: {conteos_grupos.sort_index().to_dict()}")


if __name__ == "__main__":
    main()