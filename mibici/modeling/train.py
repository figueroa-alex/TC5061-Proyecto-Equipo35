"""
Script de entrenamiento del experimento base para P1, P2 y P3 con MLFlow Tracking.
Ejecuta un baseline y un modelo base para cada pregunta, utiliza el módulo
mibici/plots.py para generar las gráficas en reports/figures/ y registra
parámetros, métricas con prefijo, etiquetas y artefactos en MLFlow.

Cómo ejecutar este script desde la raíz del proyecto:
    python -m mibici.modeling.train
"""

import json
import subprocess

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
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
from sklearn.preprocessing import StandardScaler

# Importamos nuestros tres módulos del paquete mibici (incluyendo plots)
from mibici import config, features, plots


# ==============================================================================
# FUNCIÓN AUXILIAR: OBTENER EL HASH DEL COMMIT ACTUAL DE GIT Y VERSIÓN DE DATOS
# ==============================================================================
def obtener_metadatos_trazabilidad() -> tuple[str, str]:
    """
    Obtiene el hash corto del commit actual de Git y la versión SHA-256 corta
    del archivo data_version.json para registrarlos en MLFlow.
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
# FLUJO PRINCIPAL DE ENTRENAMIENTO Y REGISTRO EN MLFLOW
# ==============================================================================
def main() -> None:
    # Creamos la carpeta de figuras si no existe
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Obtenemos el commit de Git y la versión SHA-256 de los datos
    git_commit, data_version = obtener_metadatos_trazabilidad()

    # Configuramos MLFlow Tracking local (base de datos SQLite en la raíz)
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)

    print("1. Cargando archivo Parquet limpio de MiBici...")
    columnas_necesarias = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]
    df_viajes = pd.read_parquet(config.PARQUET_FILE, columns=columnas_necesarias)
    print(f"   Viajes cargados: {len(df_viajes):,}")

    print("2. Construyendo tabla horaria por estación y variables para P1 y P2...")
    tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes)
    df_p1_p2 = features.generar_features_p1_y_p2(tabla_horaria)
    print(f"   Filas totales en tabla Estación x Hora con features: {len(df_p1_p2):,}")

    # ==========================================================================
    # PARTICIÓN ESTRICTAMENTE TEMPORAL (SIN FUGA DE INFORMACIÓN DEL FUTURO)
    # ==========================================================================
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

    print("   Partición temporal (Estación x Hora):")
    print(f"     - Train (Oct 2022 - Dic 2025): {len(df_train):,} filas")
    print(f"     - Val   (Ene 2026 - May 2026): {len(df_val):,} filas")
    print(f"     - Test  (Jun 2026 - Sep 2026): {len(df_test):,} filas")

    # Lista de columnas predictoras para P1 y P2 (ninguna usa información de la hora t)
    columnas_x_p1_p2 = [
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

    X_train = df_train[columnas_x_p1_p2]
    X_val = df_val[columnas_x_p1_p2]
    X_test = df_test[columnas_x_p1_p2]

    # ==========================================================================
    # PREGUNTA 1 (REGRESIÓN): PREDICCIÓN DE VIAJES INICIADOS POR ESTACIÓN Y HORA
    # ==========================================================================
    print("\n3. Ejecutando Experimento Base para P1 (Regresión)...")
    y_train_p1 = df_train["viajes_iniciados"]
    y_val_p1 = df_val["viajes_iniciados"]
    y_test_p1 = df_test["viajes_iniciados"]

    # Baseline simple: predecir la media del conjunto de entrenamiento
    baseline_p1 = DummyRegressor(strategy="mean")
    baseline_p1.fit(X_train, y_train_p1)
    pred_val_base_p1 = baseline_p1.predict(X_val)
    pred_test_base_p1 = baseline_p1.predict(X_test)

    # Modelo base de ML: Regresión Lineal con regularización L2 (Ridge)
    modelo_p1 = Ridge(alpha=1.0, random_state=config.RANDOM_SEED)
    modelo_p1.fit(X_train, y_train_p1)
    pred_train_p1 = np.clip(modelo_p1.predict(X_train), 0, None)
    pred_val_p1 = np.clip(modelo_p1.predict(X_val), 0, None)
    pred_test_p1 = np.clip(modelo_p1.predict(X_test), 0, None)

    metricas_p1 = {
        "baseline_val_mae": float(mean_absolute_error(y_val_p1, pred_val_base_p1)),
        "baseline_test_mae": float(mean_absolute_error(y_test_p1, pred_test_base_p1)),
        "baseline_test_rmse": float(
            np.sqrt(mean_squared_error(y_test_p1, pred_test_base_p1))
        ),
        "train_mae": float(mean_absolute_error(y_train_p1, pred_train_p1)),
        "val_mae": float(mean_absolute_error(y_val_p1, pred_val_p1)),
        "val_rmse": float(np.sqrt(mean_squared_error(y_val_p1, pred_val_p1))),
        "val_r2": float(r2_score(y_val_p1, pred_val_p1)),
        "test_mae": float(mean_absolute_error(y_test_p1, pred_test_p1)),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test_p1, pred_test_p1))),
        "test_r2": float(r2_score(y_test_p1, pred_test_p1)),
    }

    # Usamos mibici/plots.py para generar y guardar la gráfica de P1
    ruta_fig_p1 = plots.graficar_real_vs_predicho_p1(
        y_real=y_test_p1,
        y_pred=pred_test_p1,
        ruta_salida=config.FIGURES_DIR / "p1_regresion_real_vs_predicho.png",
        semilla=config.RANDOM_SEED,
    )

    with mlflow.start_run(run_name="P1_Regresion_Ridge_Base"):
        mlflow.set_tags(
            {
                "equipo": config.TEAM_TAG,
                "pregunta": "P1_Regresion",
                "tipo_modelo": "Ridge_Regression",
                "git_commit": git_commit,
            }
        )
        mlflow.log_params(
            {
                "model_type": "Ridge",
                "alpha": 1.0,
                "baseline_strategy": "DummyRegressor_mean",
                "date_range": config.DATE_RANGE_LABEL,
                "data_version": data_version,
                "random_seed": config.RANDOM_SEED,
                "granularity": "estacion_hora",
                "train_end_date": config.TRAIN_END_DATE,
                "val_end_date": config.VAL_END_DATE,
            }
        )
        mlflow.log_metrics(metricas_p1)
        mlflow.log_artifact(str(ruta_fig_p1))

    # ==========================================================================
    # PREGUNTA 2 (CLASIFICACIÓN): RIESGO DE DESBALANCE EN ESTACIÓN POR HORA
    # ==========================================================================
    print("4. Ejecutando Experimento Base para P2 (Clasificación)...")
    y_train_p2 = df_train["en_riesgo"]
    y_val_p2 = df_val["en_riesgo"]
    y_test_p2 = df_test["en_riesgo"]

    # Escalamos las variables numéricas (ajustando el escalador SOLO con Train)
    escalador_p2 = StandardScaler()
    X_train_scaled = escalador_p2.fit_transform(X_train)
    X_val_scaled = escalador_p2.transform(X_val)
    X_test_scaled = escalador_p2.transform(X_test)

    # Baseline simple: clasificador estratificado que respeta la proporción de clases
    baseline_p2 = DummyClassifier(
        strategy="stratified", random_state=config.RANDOM_SEED
    )
    baseline_p2.fit(X_train_scaled, y_train_p2)
    pred_test_base_p2 = baseline_p2.predict(X_test_scaled)

    # Modelo base de ML: Regresión Logística con balanceo de clases (class_weight='balanced')
    modelo_p2 = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        random_state=config.RANDOM_SEED,
    )
    modelo_p2.fit(X_train_scaled, y_train_p2)
    pred_val_p2 = modelo_p2.predict(X_val_scaled)
    pred_test_p2 = modelo_p2.predict(X_test_scaled)
    prob_val_p2 = modelo_p2.predict_proba(X_val_scaled)[:, 1]
    prob_test_p2 = modelo_p2.predict_proba(X_test_scaled)[:, 1]

    metricas_p2 = {
        "positive_class_ratio_train": float(y_train_p2.mean()),
        "baseline_test_f1": float(
            f1_score(y_test_p2, pred_test_base_p2, zero_division=0)
        ),
        "val_f1": float(f1_score(y_val_p2, pred_val_p2, zero_division=0)),
        "val_precision": float(precision_score(y_val_p2, pred_val_p2, zero_division=0)),
        "val_recall": float(recall_score(y_val_p2, pred_val_p2, zero_division=0)),
        "val_roc_auc": float(roc_auc_score(y_val_p2, prob_val_p2)),
        "test_f1": float(f1_score(y_test_p2, pred_test_p2, zero_division=0)),
        "test_precision": float(
            precision_score(y_test_p2, pred_test_p2, zero_division=0)
        ),
        "test_recall": float(recall_score(y_test_p2, pred_test_p2, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test_p2, prob_test_p2)),
    }

    # Usamos mibici/plots.py para generar y guardar la Matriz de Confusión de P2
    ruta_fig_p2 = plots.graficar_matriz_confusion_p2(
        y_real=y_test_p2,
        y_pred=pred_test_p2,
        ruta_salida=config.FIGURES_DIR / "p2_clasificacion_matriz_confusion.png",
    )

    with mlflow.start_run(run_name="P2_Clasificacion_LogisticRegression_Base"):
        mlflow.set_tags(
            {
                "equipo": config.TEAM_TAG,
                "pregunta": "P2_Clasificacion",
                "tipo_modelo": "LogisticRegression_Balanced",
                "git_commit": git_commit,
            }
        )
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": 200,
                "risk_capacity_threshold": config.RISK_CAPACITY_RATIO_THRESHOLD,
                "date_range": config.DATE_RANGE_LABEL,
                "data_version": data_version,
                "random_seed": config.RANDOM_SEED,
                "granularity": "estacion_hora",
            }
        )
        mlflow.log_metrics(metricas_p2)
        mlflow.log_artifact(str(ruta_fig_p2))

    # ==========================================================================
    # PREGUNTA 3 (CLUSTERING): AGRUPACIÓN DE ESTACIONES POR PATRÓN DE USO
    # ==========================================================================
    print("5. Ejecutando Experimento Base para P3 (Clustering de Estaciones)...")
    viajes_train = df_viajes.loc[
        df_viajes["Inicio_del_viaje"] <= fecha_corte_train
    ]
    df_p3 = features.generar_features_p3_clustering(viajes_train)

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

    escalador_p3 = StandardScaler()
    X_p3_scaled = escalador_p3.fit_transform(df_p3[columnas_x_p3])

    # Baseline simple: K-Means con K=2 (división mínima binaria)
    kmeans_base = KMeans(n_clusters=2, n_init=10, random_state=config.RANDOM_SEED)
    etiquetas_k2 = kmeans_base.fit_predict(X_p3_scaled)

    # Modelo base propuesto: K-Means con K=4 perfiles urbanos de estaciones
    kmeans_p3 = KMeans(n_clusters=4, n_init=10, random_state=config.RANDOM_SEED)
    etiquetas_k4 = kmeans_p3.fit_predict(X_p3_scaled)
    df_p3["cluster"] = etiquetas_k4

    metricas_p3 = {
        "baseline_k2_train_silhouette": float(
            silhouette_score(X_p3_scaled, etiquetas_k2)
        ),
        "baseline_k2_train_davies_bouldin": float(
            davies_bouldin_score(X_p3_scaled, etiquetas_k2)
        ),
        "train_silhouette": float(silhouette_score(X_p3_scaled, etiquetas_k4)),
        "train_davies_bouldin": float(
            davies_bouldin_score(X_p3_scaled, etiquetas_k4)
        ),
        "train_inertia": float(kmeans_p3.inertia_),
        "stations_clustered": float(len(df_p3)),
    }

    # Usamos mibici/plots.py para generar y guardar el mapa geográfico de P3
    ruta_fig_p3 = plots.graficar_mapa_clusters_p3(
        df_p3=df_p3,
        ruta_salida=config.FIGURES_DIR / "p3_clustering_mapa_estaciones.png",
    )

    with mlflow.start_run(run_name="P3_Clustering_KMeans_K4_Base"):
        mlflow.set_tags(
            {
                "equipo": config.TEAM_TAG,
                "pregunta": "P3_Clustering",
                "tipo_modelo": "KMeans_Clustering",
                "git_commit": git_commit,
            }
        )
        mlflow.log_params(
            {
                "model_type": "KMeans",
                "n_clusters": 4,
                "baseline_n_clusters": 2,
                "n_init": 10,
                "date_range": config.DATE_RANGE_LABEL,
                "data_version": data_version,
                "random_seed": config.RANDOM_SEED,
                "granularity": "estacion",
            }
        )
        mlflow.log_metrics(metricas_p3)
        mlflow.log_artifact(str(ruta_fig_p3))

    # ==========================================================================
    # IMPRIMIR REPORTE DE RESULTADOS EN TERMINAL
    # ==========================================================================
    print("\n" + "=" * 65)
    print("RESULTADOS DEL EXPERIMENTO BASE REGISTRADOS EN MLFLOW (EQUIPO 35)")
    print("=" * 65)
    print("P1 - REGRESIÓN (Viajes iniciados por estación y hora):")
    for k, v in metricas_p1.items():
        print(f"  {k}: {v:.4f}")
    print("\nP2 - CLASIFICACIÓN (Riesgo de desbalance >= 30% capacidad):")
    for k, v in metricas_p2.items():
        print(f"  {k}: {v:.4f}")
    print("\nP3 - CLUSTERING (Agrupación de estaciones K-Means K=4):")
    for k, v in metricas_p3.items():
        print(f"  {k}: {v:.4f}")
    print("\nConteo de estaciones por Cluster (P3):")
    print(df_p3["cluster"].value_counts().sort_index().to_dict())
    print("=" * 65)


if __name__ == "__main__":
    main()