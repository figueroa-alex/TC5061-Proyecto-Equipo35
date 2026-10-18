"""
Módulo de ingeniería de características (Feature Engineering) del proyecto MiBici.
Contiene funciones lineales, comentadas paso a paso y reutilizables para construir
las tablas de variables de las tres preguntas del proyecto (P1, P2 y P3),
garantizando que NO exista fuga de información del futuro (data leakage).
"""

import numpy as np
import pandas as pd

from mibici import config


# ==============================================================================
# FUNCIÓN AUXILIAR: CARGAR CATÁLOGO DE NOMENCLATURA DE ESTACIONES
# ==============================================================================
def cargar_nomenclatura() -> pd.DataFrame:
    """
    Lee el archivo CSV de nomenclatura de estaciones (nomenclatura_2026_09.csv)
    y devuelve un DataFrame limpio con id, name, latitude, longitude y dpcapacity.
    """
    ruta = config.NOMENCLATURE_FILE
    if not ruta.exists():
        ruta = config.RAW_DATA_DIR / "nomenclatura_2026_09.csv"

    df_nom = pd.read_csv(ruta, encoding="utf-8-sig")
    df_nom = df_nom.rename(columns={"id": "Estacion_Id"})
    df_nom["Estacion_Id"] = df_nom["Estacion_Id"].astype("int64")
    return df_nom[["Estacion_Id", "name", "latitude", "longitude", "dpcapacity"]]


# ==============================================================================
# FUNCIÓN 1: CONSTRUIR TABLA BASE HORARIA POR ESTACIÓN (PARA P1 Y P2)
# ==============================================================================
def construir_tabla_horaria_estaciones(df_viajes: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el DataFrame de viajes limpios y cuenta cuántos viajes se iniciaron
    (salidas) y cuántos terminaron (llegadas) en cada estación y en cada hora.

    Justificación:
    Tanto P1 (predicción de demanda horaria) como P2 (riesgo de desbalance respecto
    a la capacidad de la estación) operan a nivel de Estación x Hora.
    """
    # 1. Redondeamos la fecha de inicio hacia abajo a la hora cerrada (ej. 08:45 -> 08:00)
    hora_inicio = df_viajes["Inicio_del_viaje"].dt.floor("h")
    salidas = (
        df_viajes.assign(Fecha_Hora=hora_inicio)
        .groupby(["Origen_Id", "Fecha_Hora"], as_index=False)
        .size()
        .rename(columns={"Origen_Id": "Estacion_Id", "size": "viajes_iniciados"})
    )

    # 2. Redondeamos la fecha de fin hacia abajo a la hora cerrada para contar llegadas
    hora_fin = df_viajes["Fin_del_viaje"].dt.floor("h")
    llegadas = (
        df_viajes.assign(Fecha_Hora=hora_fin)
        .groupby(["Destino_Id", "Fecha_Hora"], as_index=False)
        .size()
        .rename(columns={"Destino_Id": "Estacion_Id", "size": "viajes_terminados"})
    )

    # 3. Unimos salidas y llegadas por Estacion_Id y Fecha_Hora
    tabla_horaria = pd.merge(
        salidas,
        llegadas,
        on=["Estacion_Id", "Fecha_Hora"],
        how="outer",
    )

    # 4. Si en una hora hubo salidas pero no llegadas (o viceversa), rellenamos con 0
    tabla_horaria["viajes_iniciados"] = (
        tabla_horaria["viajes_iniciados"].fillna(0).astype("int32")
    )
    tabla_horaria["viajes_terminados"] = (
        tabla_horaria["viajes_terminados"].fillna(0).astype("int32")
    )
    tabla_horaria["Estacion_Id"] = tabla_horaria["Estacion_Id"].astype("int64")

    # 5. Cruzamos con el catálogo de nomenclatura para obtener capacidad y ubicación
    df_nom = cargar_nomenclatura()
    tabla_horaria = pd.merge(
        tabla_horaria,
        df_nom[["Estacion_Id", "latitude", "longitude", "dpcapacity"]],
        on="Estacion_Id",
        how="inner",
    )

    # 6. Ordenamos estrictamente por estación y por fecha/hora de forma ascendente
    # Esto es indispensable para que los rezagos (.shift) solo miren hacia el pasado
    tabla_horaria = tabla_horaria.sort_values(
        ["Estacion_Id", "Fecha_Hora"]
    ).reset_index(drop=True)

    return tabla_horaria


# ==============================================================================
# FUNCIÓN 2: GENERAR VARIABLES PARA P1 (REGRESIÓN) Y P2 (CLASIFICACIÓN)
# ==============================================================================
def generar_features_p1_y_p2(tabla_horaria: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de la tabla horaria por estación, crea las variables predictoras
    temporales, de rezago (lags) y de capacidad para responder:
      - P1 (Regresión): predecir 'viajes_iniciados' en la hora t.
      - P2 (Clasificación): predecir 'en_riesgo' (0 o 1) en la hora t.

    Garantía contra fuga de información (Data Leakage):
    Todas las variables históricas (rezagos y promedios móviles) se calculan
    aplicando primero .shift(1) dentro de cada estación, asegurando que para
    predecir la hora t solo se utilice información ocurrida en t-1 o antes.
    """
    df = tabla_horaria.copy()

    # 1. Variables de calendario extraídas de Fecha_Hora
    df["hora"] = df["Fecha_Hora"].dt.hour.astype("int16")
    df["dia_semana"] = df["Fecha_Hora"].dt.dayofweek.astype("int16")
    df["mes"] = df["Fecha_Hora"].dt.month.astype("int16")
    df["es_fin_de_semana"] = (df["dia_semana"] >= 5).astype("int8")
    df["es_hora_pico"] = (
        df["hora"].isin([7, 8, 9, 17, 18, 19])
    ).astype("int8")

    # 2. Calculamos el flujo neto en cada hora (salidas menos llegadas)
    df["flujo_neto"] = df["viajes_iniciados"] - df["viajes_terminados"]

    # 3. Variables de rezago (lags) por estación usando ÚNICAMENTE el pasado (.shift)
    grupo_estacion = df.groupby("Estacion_Id", sort=False)

    # Viajes iniciados en la observación previa (t-1), hace 24 obs y hace 168 obs
    df["lag_1_iniciados"] = grupo_estacion["viajes_iniciados"].shift(1)
    df["lag_2_iniciados"] = grupo_estacion["viajes_iniciados"].shift(2)
    df["lag_24_iniciados"] = grupo_estacion["viajes_iniciados"].shift(24)

    # Promedio de viajes iniciados en las 3 observaciones previas (t-1, t-2 y t-3)
    df["media_movil_3_iniciados"] = (
        df["lag_1_iniciados"]
        + df["lag_2_iniciados"]
        + grupo_estacion["viajes_iniciados"].shift(3)
    ) / 3.0

    # Variables pasadas de llegadas y desbalance para P2 (nunca usan la hora actual t)
    df["lag_1_terminados"] = grupo_estacion["viajes_terminados"].shift(1)
    df["lag_1_flujo_neto"] = grupo_estacion["flujo_neto"].shift(1)
    df["lag_1_desbalance_abs"] = df["lag_1_flujo_neto"].abs()

    # Proporción del desbalance previo respecto a la capacidad de la estación
    capacidad_segura = df["dpcapacity"].replace(0, np.nan)
    df["lag_1_ratio_capacidad"] = df["lag_1_desbalance_abs"] / capacidad_segura

    # 4. Definición de la variable objetivo para P2 (Clasificación): 'en_riesgo'
    # Una estación entra en riesgo (1) en la hora t si la magnitud de su desbalance
    # |viajes_iniciados - viajes_terminados| es >= 30% de su capacidad (dpcapacity).
    desbalance_actual_abs = df["flujo_neto"].abs()
    ratio_actual = desbalance_actual_abs / capacidad_segura
    df["en_riesgo"] = (
        ratio_actual >= config.RISK_CAPACITY_RATIO_THRESHOLD
    ).astype("int8")

    # 5. Eliminamos las primeras filas de cada estación que tienen NA por los rezagos
    # y las estaciones con capacidad 0 en nomenclatura
    columnas_requeridas = [
        "lag_1_iniciados",
        "lag_2_iniciados",
        "lag_24_iniciados",
        "media_movil_3_iniciados",
        "lag_1_terminados",
        "lag_1_flujo_neto",
        "lag_1_ratio_capacidad",
    ]
    df = df.dropna(subset=columnas_requeridas).reset_index(drop=True)

    return df


# ==============================================================================
# FUNCIÓN 3: GENERAR PERFIL POR ESTACIÓN PARA P3 (CLUSTERING)
# ==============================================================================
def generar_features_p3_clustering(df_viajes_entrenamiento: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla resumen donde cada fila es una estación de MiBici y cada
    columna describe su patrón de uso habitual y su ubicación geográfica.

    Garantía contra fuga de información:
    Recibe únicamente los viajes del conjunto de entrenamiento (hasta TRAIN_END_DATE)
    para evitar usar datos futuros al definir los grupos de estaciones.
    """
    df = df_viajes_entrenamiento.copy()

    # 1. Calculamos variables auxiliares de cada viaje
    df["duracion_min"] = (
        df["Fin_del_viaje"] - df["Inicio_del_viaje"]
    ).dt.total_seconds() / 60.0
    df["hora_inicio"] = df["Inicio_del_viaje"].dt.hour
    df["es_fin_semana"] = (df["Inicio_del_viaje"].dt.dayofweek >= 5).astype("float64")
    df["es_pico_manana"] = df["hora_inicio"].isin([7, 8, 9]).astype("float64")
    df["es_pico_tarde"] = df["hora_inicio"].isin([17, 18, 19]).astype("float64")

    # Número total de días en el conjunto de entrenamiento para calcular el promedio diario
    dias_totales = max(
        1,
        (
            df["Inicio_del_viaje"].max().floor("D")
            - df["Inicio_del_viaje"].min().floor("D")
        ).days
        + 1,
    )

    # 2. Resumen de viajes iniciados por estación de origen
    resumen_origen = (
        df.groupby("Origen_Id", as_index=False)
        .agg(
            total_salidas=("Viaje_Id", "count"),
            duracion_mediana_min=("duracion_min", "median"),
            prop_fin_de_semana=("es_fin_semana", "mean"),
            prop_pico_manana=("es_pico_manana", "mean"),
            prop_pico_tarde=("es_pico_tarde", "mean"),
        )
        .rename(columns={"Origen_Id": "Estacion_Id"})
    )
    resumen_origen["promedio_salidas_diarias"] = (
        resumen_origen["total_salidas"] / float(dias_totales)
    )

    # 3. Resumen de viajes terminados por estación de destino
    resumen_destino = (
        df.groupby("Destino_Id", as_index=False)
        .agg(total_llegadas=("Viaje_Id", "count"))
        .rename(columns={"Destino_Id": "Estacion_Id"})
    )

    # 4. Unimos salidas, llegadas y catálogo de nomenclatura
    perfil_estaciones = pd.merge(
        resumen_origen, resumen_destino, on="Estacion_Id", how="inner"
    )
    df_nom = cargar_nomenclatura()
    perfil_estaciones = pd.merge(
        perfil_estaciones, df_nom, on="Estacion_Id", how="inner"
    )

    # 5. Calculamos la relación entre salidas y llegadas totales en la estación
    # Un valor > 1 indica estación principalmente emisora; < 1 indica receptora
    perfil_estaciones["ratio_salidas_llegadas"] = perfil_estaciones[
        "total_salidas"
    ] / perfil_estaciones["total_llegadas"].replace(0, 1)

    perfil_estaciones = perfil_estaciones.sort_values("Estacion_Id").reset_index(
        drop=True
    )
    return perfil_estaciones