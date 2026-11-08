"""
Servicio API REST construido con FastAPI para servir los modelos del proyecto MiBici (Equipo 35).

Expone:
  - GET  /                    -> Información general del servicio REST.
  - GET  /health              -> Estado de salud del contenedor y modelos cargados.
  - GET  /metadata            -> Metadatos de trazabilidad del Model Registry y variables requeridas.
  - POST /predict/p1-demanda  -> Predicción de demanda horaria (Regresión - P1).
  - POST /predict/p2-riesgo   -> Predicción de riesgo de desbalance horario (Clasificación - P2).
  - POST /predict/p3-grupos   -> Asignación de grupo operativo de estación (Clustering - P3).

Cómo ejecutar localmente:
    uvicorn mibici.api.main:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
import pandas as pd

from mibici import config
from mibici.api import model_loader, schemas

# ==============================================================================
# 1. VARIABLES Y ORDEN EXACTO DE COLUMNAS QUE ESPERAN LOS MODELOS DE MLFLOW
# ==============================================================================
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

COLUMNAS_P3 = [
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

# Diccionario en memoria donde viven los 3 modelos mientras la API está encendida
MODELOS_EN_MEMORIA: dict[str, dict] = {}


def asegurar_modelos_cargados() -> dict[str, dict]:
    """
    Verifica si los 3 modelos ya están cargados en memoria; si aún no lo están,
    los recupera usando la estrategia híbrida de model_loader.py.
    """
    if not MODELOS_EN_MEMORIA:
        catalogo = model_loader.cargar_todos_los_modelos()
        MODELOS_EN_MEMORIA.update(catalogo)
    return MODELOS_EN_MEMORIA


# ==============================================================================
# 2. CICLO DE VIDA DE LA APLICACIÓN (CARGA AL INICIAR Y LIMPIEZA AL APAGAR)
# ==============================================================================
@asynccontextmanager
async def ciclo_de_vida_api(app: FastAPI):
    """Carga los modelos del Model Registry una sola vez al arrancar el servidor."""
    asegurar_modelos_cargados()
    yield
    MODELOS_EN_MEMORIA.clear()


# ==============================================================================
# 3. DEFINICIÓN DE LA APLICACIÓN FASTAPI
# ==============================================================================
app = FastAPI(
    title="API REST Proyecto Integrador MiBici - Equipo 35",
    description=(
        "Servicio de inferencia para los modelos de Regresión (P1), Clasificación (P2) "
        "y Clustering (P3) del sistema de bicicletas públicas MiBici en Guadalajara, "
        "recuperados desde el Model Registry de MLflow (@champion)."
    ),
    version="1.0.0",
    lifespan=ciclo_de_vida_api,
)


# ==============================================================================
# 4. ENDPOINTS OPERATIVOS (BIENVENIDA, SALUD Y METADATOS DE TRAZABILIDAD)
# ==============================================================================
@app.get("/", tags=["Operación"])
def raiz_servicio() -> dict:
    """Endpoint de bienvenida que describe las rutas disponibles en la API."""
    return {
        "servicio": "API REST Proyecto MiBici - Equipo 35",
        "version": "1.0.0",
        "documentacion_interactiva": "/docs",
        "endpoints_prediccion": {
            "p1_demanda_horaria": "/predict/p1-demanda",
            "p2_riesgo_desbalance": "/predict/p2-riesgo",
            "p3_clustering_estaciones": "/predict/p3-grupos",
        },
    }


@app.get("/health", response_model=schemas.RespuestaSalud, tags=["Operación"])
def verificar_salud() -> schemas.RespuestaSalud:
    """Comprueba que el contenedor está activo y que los modelos se cargaron sin errores."""
    catalogo = asegurar_modelos_cargados()
    nombres_cargados = [
        info["metadata"]["model_name"] for info in catalogo.values()
    ]
    return schemas.RespuestaSalud(
        status="ok",
        team=config.TEAM_TAG,
        tracking_uri_configurada=config.MLFLOW_TRACKING_URI,
        modelos_cargados=nombres_cargados,
    )


@app.get("/metadata", tags=["Operación"])
def consultar_metadatos() -> dict:
    """Devuelve el linaje del Model Registry y las variables que espera cada modelo."""
    catalogo = asegurar_modelos_cargados()
    return {
        "team": config.TEAM_TAG,
        "load_mode": config.MODEL_LOAD_MODE,
        "models": {
            "p1": {
                **catalogo["p1"]["metadata"],
                "input_features": COLUMNAS_P1_P2,
            },
            "p2": {
                **catalogo["p2"]["metadata"],
                "input_features": COLUMNAS_P1_P2,
            },
            "p3": {
                **catalogo["p3"]["metadata"],
                "input_features": COLUMNAS_P3,
            },
        },
    }


# ==============================================================================
# 5. ENDPOINTS DE PREDICCIÓN (P1 REGRESIÓN, P2 CLASIFICACIÓN Y P3 CLUSTERING)
# ==============================================================================
@app.post(
    "/predict/p1-demanda",
    response_model=schemas.RespuestaPrediccionP1,
    tags=["Predicción MiBici"],
)
def predecir_demanda_p1(
    datos: schemas.EntradaHorariaEstacion,
) -> schemas.RespuestaPrediccionP1:
    """
    Recibe las 15 variables de una estación en una hora determinada, ejecuta el modelo
    campeón de P1 (Regresión) y devuelve el número estimado de viajes iniciados.
    """
    catalogo = asegurar_modelos_cargados()
    registro_p1 = catalogo["p1"]
    modelo_p1 = registro_p1["model"]
    meta_p1 = registro_p1["metadata"]

    # Convertimos los datos validados por Pydantic en un DataFrame de 1 fila en float64
    fila_dict = datos.model_dump()
    df_entrada = pd.DataFrame([fila_dict])[COLUMNAS_P1_P2].astype("float64")

    # Ejecutamos la inferencia y aseguramos que la demanda predicha no sea negativa
    resultado_bruto = modelo_p1.predict(df_entrada)
    valor_predicho = max(0.0, float(resultado_bruto[0]))

    return schemas.RespuestaPrediccionP1(
        question="p1",
        model_name=meta_p1["model_name"],
        model_version=str(meta_p1["model_version"]),
        model_alias=meta_p1["model_alias"],
        loaded_from=meta_p1["loaded_from"],
        viajes_iniciados_predichos=round(valor_predicho, 4),
    )


@app.post(
    "/predict/p2-riesgo",
    response_model=schemas.RespuestaPrediccionP2,
    tags=["Predicción MiBici"],
)
def predecir_riesgo_p2(
    datos: schemas.EntradaHorariaEstacion,
) -> schemas.RespuestaPrediccionP2:
    """
    Recibe las 15 variables de una estación en una hora determinada, ejecuta el modelo
    campeón de P2 (Clasificación binaria) y devuelve si estará en riesgo de desbalance.
    """
    catalogo = asegurar_modelos_cargados()
    registro_p2 = catalogo["p2"]
    modelo_p2 = registro_p2["model"]
    meta_p2 = registro_p2["metadata"]

    fila_dict = datos.model_dump()
    df_entrada = pd.DataFrame([fila_dict])[COLUMNAS_P1_P2].astype("float64")

    resultado_bruto = modelo_p2.predict(df_entrada)
    clase_predicha = int(resultado_bruto[0])
    etiqueta = "EN_RIESGO" if clase_predicha == 1 else "SIN_RIESGO"

    return schemas.RespuestaPrediccionP2(
        question="p2",
        model_name=meta_p2["model_name"],
        model_version=str(meta_p2["model_version"]),
        model_alias=meta_p2["model_alias"],
        loaded_from=meta_p2["loaded_from"],
        en_riesgo=clase_predicha,
        etiqueta_riesgo=etiqueta,
    )


@app.post(
    "/predict/p3-grupos",
    response_model=schemas.RespuestaPrediccionP3,
    tags=["Predicción MiBici"],
)
def predecir_grupo_p3(
    datos: schemas.EntradaPerfilEstacionP3,
) -> schemas.RespuestaPrediccionP3:
    """
    Recibe las 9 variables del perfil histórico y geográfico de una estación, ejecuta
    el modelo campeón de P3 (Clustering) y devuelve el grupo operativo asignado.
    """
    catalogo = asegurar_modelos_cargados()
    registro_p3 = catalogo["p3"]
    modelo_p3 = registro_p3["model"]
    meta_p3 = registro_p3["metadata"]

    fila_dict = datos.model_dump()
    df_entrada = pd.DataFrame([fila_dict])[COLUMNAS_P3].astype("float64")

    resultado_bruto = modelo_p3.predict(df_entrada)
    cluster_id = int(resultado_bruto[0])

    return schemas.RespuestaPrediccionP3(
        question="p3",
        model_name=meta_p3["model_name"],
        model_version=str(meta_p3["model_version"]),
        model_alias=meta_p3["model_alias"],
        loaded_from=meta_p3["loaded_from"],
        cluster_asignado=cluster_id,
        descripcion_cluster=f"Grupo operativo {cluster_id} de estaciones MiBici en el AMG.",
    )