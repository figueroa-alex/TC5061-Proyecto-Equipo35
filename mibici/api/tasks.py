"""
Tareas de inferencia asíncrona ejecutadas en segundo plano por el worker de RQ (Equipo 35).

Flujo de trabajo:
  1. La API FastAPI recibe una solicitud en POST /predict/async/p1-demanda.
  2. La API valida los datos con Pydantic y encola la función
     'ejecutar_prediccion_p1_asincrona' dentro de Redis usando RQ.
  3. El contenedor 'rq-worker' toma el trabajo de la cola, carga el modelo
     campeón de P1 (si aún no está en memoria del worker), ejecuta la predicción
     y guarda el resultado en Redis para que el cliente lo consulte en GET /jobs/{job_id}.
"""

import pandas as pd

from mibici.api import model_loader

# Orden exacto de las 15 variables predictoras que espera el modelo P1 (Demanda horaria)
COLUMNAS_P1_P2 = [
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

# Caché simple en memoria del worker para no recargar el modelo del disco en cada trabajo
_CACHE_MODELO_P1: dict = {}


def ejecutar_prediccion_p1_asincrona(datos_entrada: dict) -> dict:
    """
    Ejecuta en segundo plano la predicción de demanda horaria (P1) dentro del worker de RQ.

    Parámetros:
        datos_entrada (dict): Diccionario ya validado con las 15 variables de la estación.

    Retorna:
        dict: Resultado estructurado con los metadatos del modelo y la demanda estimada.
    """
    # Paso 1: Cargar el modelo campeón de P1 solo la primera vez que trabaja el worker
    if "model" not in _CACHE_MODELO_P1:
        modelo_p1, metadatos_p1 = model_loader.cargar_modelo_para_api("p1")
        _CACHE_MODELO_P1["model"] = modelo_p1
        _CACHE_MODELO_P1["metadata"] = metadatos_p1

    modelo_p1 = _CACHE_MODELO_P1["model"]
    meta_p1 = _CACHE_MODELO_P1["metadata"]

    # Paso 2: Construir un DataFrame de 1 fila con las columnas en orden exacto (float64)
    df_entrada = pd.DataFrame([datos_entrada])[COLUMNAS_P1_P2].astype("float64")

    # Paso 3: Calcular la predicción y garantizar que el número de viajes no sea negativo
    resultado_bruto = modelo_p1.predict(df_entrada)
    valor_predicho = max(0.0, float(resultado_bruto[0]))

    # Paso 4: Devolver el diccionario final que RQ almacenará en Redis
    return {
        "question": "p1",
        "model_name": meta_p1["model_name"],
        "model_version": str(meta_p1["model_version"]),
        "model_alias": meta_p1["model_alias"],
        "loaded_from": meta_p1["loaded_from"],
        "viajes_iniciados_predichos": round(valor_predicho, 4),
    }