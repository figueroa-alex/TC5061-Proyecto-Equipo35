"""
Validación automatizada de calidad, integridad y drift de datos con DeepChecks (Equipo 35).
Soporta dos modos de ejecución configurables mediante la variable de entorno MIBICI_TEST_DATA_MODE:
  1. Modo 'real' (o 'auto' cuando existen los archivos Parquet): evalúa muestras
     representativas de data/processed/datos_mibici.parquet, data/interim/features_p1_p2.parquet
     y data/interim/features_p3.parquet.
  2. Modo 'synthetic' (o 'auto' en un clon limpio antes de correr la ingesta): evalúa
     el dataset sintético determinista de respaldo para que pytest siempre sea ejecutable.
"""

import os

from deepchecks.tabular import Dataset
from deepchecks.tabular.checks import (
    DataDuplicates,
    FeatureDrift,
    IsSingleValue,
    LabelDrift,
    MixedDataTypes,
    MixedNulls,
    OutlierSampleDetection,
    SpecialCharacters,
    StringMismatch,
    TrainTestSamplesMix,
)
import numpy as np
import pandas as pd

from mibici import config, features


def _obtener_modo_datos() -> str:
    """Lee el modo de datos ('auto', 'real' o 'synthetic') desde el entorno o config.py."""
    return os.getenv("MIBICI_TEST_DATA_MODE", config.TEST_DATA_MODE).lower().strip()


# ==============================================================================
# FUNCIONES AUXILIARES: CARGAR MUESTRAS REALES O SINTÉTICAS (VIAJES, P1/P2 Y P3)
# ==============================================================================
def obtener_muestra_viajes_para_deepchecks(
    df_viajes_sinteticos: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """
    Devuelve un DataFrame de viajes listo para DeepChecks y una etiqueta ('real' o 'synthetic')
    indicando qué fuente se utilizó según MIBICI_TEST_DATA_MODE y la presencia del Parquet.
    """
    modo = _obtener_modo_datos()
    usar_real = modo in ("auto", "real") and config.PARQUET_FILE.exists()

    if usar_real:
        df = pd.read_parquet(config.PARQUET_FILE).head(10000).copy()
        origen_datos = "real"
    else:
        df = df_viajes_sinteticos.copy()
        origen_datos = "synthetic"

    for col in ["Viaje_Id", "Usuario_Id", "Origen_Id", "Destino_Id"]:
        df[col] = df[col].astype("int64")
    df["Año_de_nacimiento"] = df["Año_de_nacimiento"].astype("float64")
    df["Genero"] = df["Genero"].fillna("FALTANTE").astype(object)
    return df, origen_datos


def obtener_particion_train_test_para_deepchecks(
    monkeypatch,
    df_nomenclatura_sintetica: pd.DataFrame,
    df_viajes_sinteticos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Devuelve muestras de Train y Test de la tabla de features de P1/P2 para validar
    FeatureDrift, LabelDrift (en P1 y P2) y TrainTestSamplesMix con DeepChecks.
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
    columnas_evaluadas = columnas_x + ["viajes_iniciados", "en_riesgo"]

    modo = _obtener_modo_datos()
    usar_real = modo in ("auto", "real") and config.INTERIM_P1_P2_FILE.exists()

    if usar_real:
        df_feat = pd.read_parquet(
            config.INTERIM_P1_P2_FILE,
            columns=["Fecha_Hora"] + columnas_evaluadas,
        )
        corte_train = pd.Timestamp(config.TRAIN_END_DATE)
        corte_val = pd.Timestamp(config.VAL_END_DATE)

        df_train = df_feat.loc[
            df_feat["Fecha_Hora"] <= corte_train, columnas_evaluadas
        ].sample(n=5000, random_state=config.RANDOM_SEED)
        df_test = df_feat.loc[
            df_feat["Fecha_Hora"] > corte_val, columnas_evaluadas
        ].sample(n=5000, random_state=config.RANDOM_SEED)
        return df_train.reset_index(drop=True), df_test.reset_index(drop=True), "real"

    monkeypatch.setattr(
        features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica
    )
    tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes_sinteticos)
    df_feat_completo = (
        features.generar_features_p1_y_p2(tabla_horaria)
        .sort_values(["Fecha_Hora", "Estacion_Id"])
        .reset_index(drop=True)
    )
    df_feat = df_feat_completo[columnas_evaluadas]

    mitad = len(df_feat) // 2
    df_train = df_feat.iloc[:mitad].reset_index(drop=True)
    df_test = df_feat.iloc[mitad:].reset_index(drop=True)
    return df_train, df_test, "synthetic"


def obtener_perfiles_p3_train_test_para_deepchecks(
    monkeypatch,
    df_nomenclatura_sintetica: pd.DataFrame,
    df_viajes_sinteticos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Devuelve la tabla de perfiles por estación de P3 para el periodo de entrenamiento (Train)
    y para el periodo de prueba (Test), tanto en modo real como en modo sintético.
    """
    modo = _obtener_modo_datos()
    usar_real = (
        modo in ("auto", "real")
        and config.INTERIM_P3_FILE.exists()
        and config.PARQUET_FILE.exists()
    )

    if usar_real:
        df_p3_train = pd.read_parquet(config.INTERIM_P3_FILE)
        corte_val = pd.Timestamp(config.VAL_END_DATE)
        columnas_viajes = [
            "Viaje_Id",
            "Inicio_del_viaje",
            "Fin_del_viaje",
            "Origen_Id",
            "Destino_Id",
        ]
        df_viajes_test = pd.read_parquet(
            config.PARQUET_FILE,
            columns=columnas_viajes,
            filters=[("Inicio_del_viaje", ">", corte_val)],
        )
        df_p3_test = features.generar_features_p3_clustering(df_viajes_test)
        return df_p3_train.reset_index(drop=True), df_p3_test.reset_index(drop=True), "real"

    monkeypatch.setattr(
        features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica
    )
    df_ordenado = df_viajes_sinteticos.sort_values("Inicio_del_viaje").reset_index(drop=True)
    df_p3_train = features.generar_features_p3_clustering(df_ordenado.iloc[0::2])
    df_p3_test = features.generar_features_p3_clustering(df_ordenado.iloc[1::2])
    return df_p3_train.reset_index(drop=True), df_p3_test.reset_index(drop=True), "synthetic"


# ==============================================================================
# PRUEBA 1: INTEGRIDAD DEL DATASET DE VIAJES (TIPOS, NULOS, DUPLICADOS, STRINGS)
# ==============================================================================
def test_deepchecks_integridad_dataset_viajes(df_viajes_validos_sinteticos):
    """
    Valida con DeepChecks la integridad del dataset limpio de viajes:
      1. IsSingleValue: ninguna columna tiene un único valor constante.
      2. MixedDataTypes: ninguna columna mezcla tipos de datos (ej. números y texto).
      3. MixedNulls: no coexisten distintos tipos de representaciones nulas.
      4. StringMismatch: la columna categórica 'Genero' no presenta variantes con errores.
      5. SpecialCharacters: 'Genero' no contiene caracteres especiales inesperados.
      6. Duplicados exactos: 0% al incluir el identificador del viaje.
    """
    df_viajes, origen = obtener_muestra_viajes_para_deepchecks(
        df_viajes_validos_sinteticos
    )

    ds_con_id = Dataset(
        df_viajes.drop(columns=["Inicio_del_viaje", "Fin_del_viaje"]),
        index_name="Viaje_Id",
        cat_features=["Genero", "Origen_Id", "Destino_Id"],
    )

    res_single = IsSingleValue().run(ds_con_id)
    for col, conteo_unicos in res_single.value.items():
        assert conteo_unicos > 1, (
            f"La columna {col} tiene un solo valor constante ({origen})."
        )

    res_mixed_types = MixedDataTypes().run(ds_con_id)
    for col, info_tipos in res_mixed_types.value.items():
        assert info_tipos == {}, (
            f"La columna {col} presenta tipos de datos mezclados: {info_tipos}"
        )

    res_mixed_nulls = MixedNulls().run(ds_con_id)
    for col, info_nulos in res_mixed_nulls.value["columns"].items():
        assert len(info_nulos) <= 1, (
            f"La columna {col} mezcla distintos tipos de nulos: {info_nulos}"
        )

    res_strings = StringMismatch().run(ds_con_id)
    assert res_strings.value["columns"].get("Genero", {}) == {}

    res_special = SpecialCharacters().run(ds_con_id)
    mapa_special = res_special.value.get("columns", res_special.value)
    assert float(mapa_special.get("Genero", 0.0)) == 0.0

    # Duplicados exactos considerando todas las columnas del viaje (incluyendo Viaje_Id)
    assert int(df_viajes.duplicated().sum()) == 0


# ==============================================================================
# PRUEBA 2: VALIDACIÓN DE PARTICIÓN TRAIN VS. TEST Y DRIFT EN P1 ('viajes_iniciados')
# ==============================================================================
def test_deepchecks_drift_y_fuga_train_test(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Ejecuta los chequeos Train-Test de DeepChecks sobre las variables y la etiqueta de P1:
      1. TrainTestSamplesMix: verifica que no exista filtración de muestras entre Train y Test.
      2. FeatureDrift: verifica que las variables operativas principales (dpcapacity,
         lag_1_iniciados, latitude, longitude) tengan un drift bajo (< 0.25), y documenta
         el drift estacional esperado en 'mes' cuando se evalúa la partición temporal real.
      3. LabelDrift: verifica la estabilidad de la variable objetivo 'viajes_iniciados'.
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
    columnas_cat = ["hora", "dia_semana", "mes", "es_fin_de_semana", "es_hora_pico"]

    df_train, df_test, origen = obtener_particion_train_test_para_deepchecks(
        monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
    )

    ds_train = Dataset(
        df_train[columnas_x + ["viajes_iniciados"]],
        label="viajes_iniciados",
        cat_features=columnas_cat,
    )
    ds_test = Dataset(
        df_test[columnas_x + ["viajes_iniciados"]],
        label="viajes_iniciados",
        cat_features=columnas_cat,
    )

    # 1. Chequeo de mezcla de muestras entre entrenamiento y prueba
    res_mix = TrainTestSamplesMix().run(ds_train, ds_test)
    assert res_mix.value["ratio"] <= 0.05

    # 2. Chequeo de Feature Drift
    res_feature_drift = FeatureDrift().run(ds_train, ds_test)
    for col_estable in ["dpcapacity", "lag_1_iniciados", "latitude", "longitude"]:
        score_drift = float(res_feature_drift.value[col_estable]["Drift score"])
        assert score_drift < 0.25, (
            f"Drift excesivo detectado en {col_estable} ({origen}): {score_drift:.4f}"
        )

    # En los datos reales, 'mes' debe reflejar el corte temporal (Oct 2022-Dic 2025 vs Jun-Sep 2026)
    if origen == "real":
        drift_mes = float(res_feature_drift.value["mes"]["Drift score"])
        assert drift_mes > 0.50

    # 3. Chequeo de Label Drift para P1 (viajes_iniciados)
    res_label_drift = LabelDrift().run(ds_train, ds_test)
    score_label_drift = float(res_label_drift.value["Drift score"])
    assert score_label_drift < 0.20, (
        f"Label drift excesivo en viajes_iniciados ({origen}): {score_label_drift:.4f}"
    )


# ==============================================================================
# PRUEBA 3: COMPARACIÓN TRAIN VS. TEST CON LA ETIQUETA DE RIESGO DE P2 ('en_riesgo')
# ==============================================================================
def test_deepchecks_drift_etiqueta_riesgo_p2(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Valida con DeepChecks la partición de entrenamiento y prueba utilizando la etiqueta
    binaria de clasificación de P2 ('en_riesgo', desbalance >= 30% de dpcapacity):
      1. TrainTestSamplesMix: ausencia de filtración de filas entre Train y Test.
      2. LabelDrift (Cramer's V sobre clasificación binaria): verifica que la proporción
         de estaciones en riesgo sea estable entre entrenamiento y prueba (< 0.20).
      3. FeatureDrift: consistencia de las variables predictoras de desbalance previo.
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
    columnas_cat = ["hora", "dia_semana", "mes", "es_fin_de_semana", "es_hora_pico"]

    df_train, df_test, origen = obtener_particion_train_test_para_deepchecks(
        monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
    )

    ds_train_p2 = Dataset(
        df_train[columnas_x + ["en_riesgo"]],
        label="en_riesgo",
        label_type="binary",
        cat_features=columnas_cat,
    )
    ds_test_p2 = Dataset(
        df_test[columnas_x + ["en_riesgo"]],
        label="en_riesgo",
        label_type="binary",
        cat_features=columnas_cat,
    )

    res_mix_p2 = TrainTestSamplesMix().run(ds_train_p2, ds_test_p2)
    assert res_mix_p2.value["ratio"] <= 0.05

    res_label_drift_p2 = LabelDrift().run(ds_train_p2, ds_test_p2)
    score_ld_p2 = float(res_label_drift_p2.value["Drift score"])
    assert score_ld_p2 < 0.20, (
        f"Label drift excesivo en etiqueta en_riesgo de P2 ({origen}): {score_ld_p2:.4f}"
    )

    res_fd_p2 = FeatureDrift().run(ds_train_p2, ds_test_p2)
    for col_desbalance in ["lag_1_terminados", "lag_1_flujo_neto", "lag_1_ratio_capacidad"]:
        score_fd = float(res_fd_p2.value[col_desbalance]["Drift score"])
        assert score_fd < 0.25, (
            f"Drift excesivo en variable de P2 {col_desbalance} ({origen}): {score_fd:.4f}"
        )


# ==============================================================================
# PRUEBA 4: INTEGRIDAD Y COMPARACIÓN TRAIN VS. TEST DE LA TABLA DE PERFILES DE P3
# ==============================================================================
def test_deepchecks_perfiles_estaciones_p3_integridad_y_drift(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Valida con DeepChecks la tabla de perfiles por estación de P3 (Clustering):
      1. Integridad del conjunto de entrenamiento (IsSingleValue, MixedNulls,
         MixedDataTypes y DataDuplicates == 0.0).
      2. Comparación entre los perfiles de estaciones calculados en entrenamiento (Train)
         y los perfiles calculados en prueba (Test) mediante FeatureDrift y TrainTestSamplesMix.
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

    df_p3_train, df_p3_test, origen = obtener_perfiles_p3_train_test_para_deepchecks(
        monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
    )

    ds_p3_train = Dataset(df_p3_train[columnas_x_p3], cat_features=[])
    ds_p3_test = Dataset(df_p3_test[columnas_x_p3], cat_features=[])

    # 1. Chequeos de integridad sobre la tabla de perfiles de P3
    res_single_p3 = IsSingleValue().run(ds_p3_train)
    for col, conteo_unicos in res_single_p3.value.items():
        assert conteo_unicos > 1, f"Columna constante en P3 ({origen}): {col}"

    res_nulls_p3 = MixedNulls().run(ds_p3_train)
    for col, info_nulos in res_nulls_p3.value["columns"].items():
        assert info_nulos == {}, f"Nulos detectados en P3 ({origen}) columna {col}: {info_nulos}"

    res_types_p3 = MixedDataTypes().run(ds_p3_train)
    for col, info_tipos in res_types_p3.value.items():
        assert info_tipos == {}, f"Tipos mezclados en P3 ({origen}) columna {col}: {info_tipos}"

    res_dup_p3 = DataDuplicates().run(ds_p3_train)
    assert float(res_dup_p3.value) == 0.0

    # 2. Comparación Train vs. Test de los perfiles de estaciones de P3
    res_mix_p3 = TrainTestSamplesMix().run(ds_p3_train, ds_p3_test)
    assert res_mix_p3.value["ratio"] <= 0.05

    res_fd_p3 = FeatureDrift(min_samples=5).run(ds_p3_train, ds_p3_test)
    if origen == "real":
        # En datos reales (373 estaciones Train vs 469 Test), las 9 variables tienen KS < 0.20
        for col in columnas_x_p3:
            score_ks = float(res_fd_p3.value[col]["Drift score"])
            assert score_ks < 0.20, (
                f"Drift excesivo en perfil P3 real columna {col}: {score_ks:.4f}"
            )
    else:
        # En modo sintético (6 estaciones), las variables estructurales del catálogo no cambian
        for col_fija in ["dpcapacity", "latitude", "longitude"]:
            score_ks = float(res_fd_p3.value[col_fija]["Drift score"])
            assert score_ks == 0.0


# ==============================================================================
# PRUEBA 5: DETECCIÓN DE PROBLEMAS INYECTADOS (CASOS LÍMITE CON DEEPCHECKS)
# ==============================================================================
def test_deepchecks_detecta_problemas_inyectados():
    """
    Caso límite: Inyecta deliberadamente filas duplicadas, tipos de datos mezclados y
    un valor atípico (outlier) extremo en un dataset de 120 muestras para comprobar que
    los chequeos de DeepChecks detectan y cuantifican cada anomalía.
    """
    estaciones = [10, 10] + [20 + (i % 5) * 10 for i in range(118)]
    generos = ["M", "M"] + ["F" if i % 2 == 0 else "M" for i in range(117)] + [999]
    viajes = [2.0, 2.0] + [float(2 + (i % 4)) for i in range(117)] + [5000.0]

    df_corrupto = pd.DataFrame(
        {
            "Estacion_Id": estaciones,
            "Genero": generos,
            "viajes_iniciados": viajes,
        }
    )
    ds_corrupto = Dataset(
        df_corrupto,
        label="viajes_iniciados",
        cat_features=["Genero"],
    )

    # 1. DataDuplicates debe detectar las filas idénticas (ratio > 0)
    res_dup = DataDuplicates().run(ds_corrupto)
    assert float(res_dup.value) > 0.0

    # 2. MixedDataTypes debe detectar que 'Genero' mezcla strings con el entero 999
    res_mixed = MixedDataTypes().run(ds_corrupto)
    assert res_mixed.value["Genero"] != {}

    # 3. OutlierSampleDetection devuelve un np.ndarray de 1000 cuantiles de probabilidad LoOP
    res_outliers = OutlierSampleDetection().run(ds_corrupto)
    assert isinstance(res_outliers.value, np.ndarray)
    assert len(res_outliers.value) == 1000
    assert float(np.max(res_outliers.value)) > 0.50