"""
Pruebas automatizadas de reproducibilidad del entrenamiento (Equipo 35).
Verifica que dos ejecuciones independientes del entrenamiento con la misma semilla
(config.RANDOM_SEED = 42) produzcan exactamente las mismas predicciones y métricas
para las tres preguntas del proyecto (P1 Regresión, P2 Clasificación y P3 Clustering).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score, silhouette_score

from mibici import config, features
from mibici.modeling import train


def _obtener_datos_entrenamiento_p1_p2(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Si existe data/interim/features_p1_p2.parquet y el modo no es 'synthetic',
    toma una muestra determinista de 2,000 filas reales para evaluar la reproducibilidad.
    De lo contrario, construye la tabla desde el dataset sintético de respaldo.
    """
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

    usar_real = (
        config.TEST_DATA_MODE in ("auto", "real")
        and config.INTERIM_P1_P2_FILE.exists()
    )
    if usar_real:
        df_muestra = pd.read_parquet(
            config.INTERIM_P1_P2_FILE,
            columns=columnas_x + ["viajes_iniciados", "en_riesgo"],
        ).head(2000)
    else:
        monkeypatch.setattr(features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica)
        tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes_validos_sinteticos)
        df_muestra = features.generar_features_p1_y_p2(tabla_horaria)

    X = df_muestra[columnas_x].astype("float64")
    y_p1 = df_muestra["viajes_iniciados"].astype("float64")
    y_p2 = df_muestra["en_riesgo"].astype("int64")
    return X, y_p1, y_p2


def test_reproducibilidad_entrenamiento_p1_regresion(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Verifica que entrenar dos veces el modelo campeón de P1 (HistGradientBoostingRegressor)
    y RandomForestRegressor con la misma semilla (RANDOM_SEED = 42) produzca exactamente
    el mismo vector de predicciones y las mismas métricas MAE y R2.
    """
    X, y_p1, _ = _obtener_datos_entrenamiento_p1_p2(
        monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
    )
    corte = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y_p1.iloc[:corte], y_p1.iloc[corte:]

    for codigo_modelo in ["hgb", "rf", "ridge"]:
        modelo_ejecucion_1 = train.construir_estimador(
            question="p1",
            stage="candidate",
            model=codigo_modelo,
            semilla=config.RANDOM_SEED,
            n_estimators=40,
            max_depth=6,
        )
        modelo_ejecucion_2 = train.construir_estimador(
            question="p1",
            stage="candidate",
            model=codigo_modelo,
            semilla=config.RANDOM_SEED,
            n_estimators=40,
            max_depth=6,
        )

        modelo_ejecucion_1.fit(X_train, y_train)
        modelo_ejecucion_2.fit(X_train, y_train)

        pred_1 = np.clip(modelo_ejecucion_1.predict(X_test), 0, None)
        pred_2 = np.clip(modelo_ejecucion_2.predict(X_test), 0, None)

        mae_1 = mean_absolute_error(y_test, pred_1)
        mae_2 = mean_absolute_error(y_test, pred_2)
        r2_1 = r2_score(y_test, pred_1)
        r2_2 = r2_score(y_test, pred_2)

        np.testing.assert_allclose(pred_1, pred_2, rtol=1e-12, atol=1e-12)
        assert mae_1 == mae_2
        assert r2_1 == r2_2


def test_reproducibilidad_entrenamiento_p2_clasificacion(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Verifica que entrenar dos veces el modelo de P2 (HistGradientBoostingClassifier y LogReg)
    con la misma semilla (RANDOM_SEED = 42) produzca exactamente las mismas clases predichas,
    probabilidades, F1-Score y ROC-AUC.
    """
    X, _, y_p2 = _obtener_datos_entrenamiento_p1_p2(
        monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
    )
    # Garantizamos que existan ambas clases (0 y 1) en entrenamiento y prueba
    if y_p2.nunique() < 2:
        y_p2 = y_p2.copy()
        y_p2.iloc[::5] = 1

    corte = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:corte], X.iloc[corte:]
    y_train, y_test = y_p2.iloc[:corte], y_p2.iloc[corte:]

    for codigo_modelo in ["hgb_clf", "logreg", "rf_clf"]:
        clf_1 = train.construir_estimador(
            question="p2",
            stage="candidate",
            model=codigo_modelo,
            semilla=config.RANDOM_SEED,
            n_estimators=40,
            max_depth=6,
        )
        clf_2 = train.construir_estimador(
            question="p2",
            stage="candidate",
            model=codigo_modelo,
            semilla=config.RANDOM_SEED,
            n_estimators=40,
            max_depth=6,
        )

        clf_1.fit(X_train, y_train)
        clf_2.fit(X_train, y_train)

        clases_1 = clf_1.predict(X_test)
        clases_2 = clf_2.predict(X_test)
        prob_1 = clf_1.predict_proba(X_test)[:, 1]
        prob_2 = clf_2.predict_proba(X_test)[:, 1]

        np.testing.assert_array_equal(clases_1, clases_2)
        np.testing.assert_allclose(prob_1, prob_2, rtol=1e-12, atol=1e-12)
        assert f1_score(y_test, clases_1, zero_division=0) == f1_score(
            y_test, clases_2, zero_division=0
        )
        if y_test.nunique() == 2:
            assert roc_auc_score(y_test, prob_1) == roc_auc_score(y_test, prob_2)


def test_reproducibilidad_entrenamiento_p3_clustering(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Verifica que dos ejecuciones de K-Means para P3 con la misma semilla (RANDOM_SEED = 42)
    asignen exactamente el mismo cluster a cada estación y obtengan el mismo Silhouette Score.
    """
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

    usar_real = (
        config.TEST_DATA_MODE in ("auto", "real")
        and config.INTERIM_P3_FILE.exists()
    )
    if usar_real:
        df_p3 = pd.read_parquet(config.INTERIM_P3_FILE)
        k_clusters = 4
    else:
        monkeypatch.setattr(features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica)
        df_p3 = features.generar_features_p3_clustering(df_viajes_validos_sinteticos)
        k_clusters = 3

    X_p3 = df_p3[columnas_x_p3].astype("float64")

    modelo_cluster_1 = train.construir_estimador(
        question="p3",
        stage="candidate",
        model="kmeans",
        semilla=config.RANDOM_SEED,
        n_clusters=k_clusters,
    )
    modelo_cluster_2 = train.construir_estimador(
        question="p3",
        stage="candidate",
        model="kmeans",
        semilla=config.RANDOM_SEED,
        n_clusters=k_clusters,
    )

    etiquetas_1 = modelo_cluster_1.fit_predict(X_p3)
    etiquetas_2 = modelo_cluster_2.fit_predict(X_p3)

    np.testing.assert_array_equal(etiquetas_1, etiquetas_2)

    X_scaled = modelo_cluster_1.named_steps["scaler"].transform(X_p3)
    assert silhouette_score(X_scaled, etiquetas_1) == silhouette_score(X_scaled, etiquetas_2)