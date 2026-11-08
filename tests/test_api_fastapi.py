"""
Pruebas automatizadas con PyTest para la API REST en FastAPI y sus esquemas Pydantic (Equipo 35).

Verifica:
  1. Endpoints operativos (/ y /health y /metadata) y carga desde el Model Registry.
  2. Predicciones válidas en formato JSON para los 3 endpoints (P1, P2 y P3).
  3. Validaciones robustas de entrada (casos límite: horas fuera de rango, capacidad cero,
     proporciones inválidas y rechazo de campos personales no permitidos con código 422).
"""

from fastapi.testclient import TestClient

from mibici import config
from mibici.api.main import app

cliente_api = TestClient(app)


# ==============================================================================
# DATOS DE PRUEBA VÁLIDOS BASADOS EN ESTACIONES REALES DE MIBICI (GDL-003)
# ==============================================================================
PAYLOAD_VALIDO_P1_P2 = {
    "hora": 8,
    "dia_semana": 1,
    "mes": 9,
    "es_fin_de_semana": 0,
    "es_hora_pico": 1,
    "dpcapacity": 19.0,
    "latitude": 20.66769,
    "longitude": -103.368252,
    "lag_1_iniciados": 6.0,
    "lag_2_iniciados": 3.0,
    "lag_24_iniciados": 5.0,
    "media_movil_3_iniciados": 4.3333,
    "lag_1_terminados": 2.0,
    "lag_1_flujo_neto": 4.0,
    "lag_1_ratio_capacidad": 0.2105,
}

PAYLOAD_VALIDO_P3 = {
    "promedio_salidas_diarias": 42.5,
    "duracion_mediana_min": 11.2,
    "prop_fin_de_semana": 0.21,
    "prop_pico_manana": 0.24,
    "prop_pico_tarde": 0.28,
    "ratio_salidas_llegadas": 1.04,
    "dpcapacity": 19.0,
    "latitude": 20.66769,
    "longitude": -103.368252,
}


# ==============================================================================
# 1. PRUEBAS DE ENDPOINTS OPERATIVOS Y RECUPERACIÓN DEL MODEL REGISTRY
# ==============================================================================
def test_api_health_y_metadata_modelos_cargados():
    """Verifica que /health y /metadata respondan 200 y reporten los 3 modelos registrados."""
    resp_health = cliente_api.get("/health")
    assert resp_health.status_code == 200
    datos_health = resp_health.json()
    assert datos_health["status"] == "ok"
    assert datos_health["team"] == config.TEAM_TAG
    assert len(datos_health["modelos_cargados"]) == 3

    resp_meta = cliente_api.get("/metadata")
    assert resp_meta.status_code == 200
    datos_meta = resp_meta.json()
    assert datos_meta["models"]["p1"]["model_name"] == config.MODEL_NAME_P1
    assert datos_meta["models"]["p2"]["model_name"] == config.MODEL_NAME_P2
    assert datos_meta["models"]["p3"]["model_name"] == config.MODEL_NAME_P3


# ==============================================================================
# 2. PRUEBAS DE PREDICCIÓN FUNCIONAL EN LOS 3 ENDPOINTS (CASOS NORMALES)
# ==============================================================================
def test_api_prediccion_p1_demanda_valida():
    """Caso normal: /predict/p1-demanda recibe datos válidos y devuelve predicción >= 0."""
    respuesta = cliente_api.post("/predict/p1-demanda", json=PAYLOAD_VALIDO_P1_P2)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["question"] == "p1"
    assert cuerpo["model_name"] == config.MODEL_NAME_P1
    assert cuerpo["model_alias"] == config.PRODUCTION_ALIAS
    assert isinstance(cuerpo["viajes_iniciados_predichos"], float)
    assert cuerpo["viajes_iniciados_predichos"] >= 0.0


def test_api_prediccion_p2_riesgo_valida():
    """Caso normal: /predict/p2-riesgo recibe datos válidos y devuelve clase 0 o 1."""
    respuesta = cliente_api.post("/predict/p2-riesgo", json=PAYLOAD_VALIDO_P1_P2)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["question"] == "p2"
    assert cuerpo["model_name"] == config.MODEL_NAME_P2
    assert cuerpo["en_riesgo"] in (0, 1)
    assert cuerpo["etiqueta_riesgo"] in ("EN_RIESGO", "SIN_RIESGO")


def test_api_prediccion_p3_grupos_valida():
    """Caso normal: /predict/p3-grupos recibe perfil válido y devuelve el cluster asignado."""
    respuesta = cliente_api.post("/predict/p3-grupos", json=PAYLOAD_VALIDO_P3)
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["question"] == "p3"
    assert cuerpo["model_name"] == config.MODEL_NAME_P3
    assert isinstance(cuerpo["cluster_asignado"], int)
    assert cuerpo["cluster_asignado"] >= 0


# ==============================================================================
# 3. PRUEBAS DE VALIDACIÓN ROBUSTA DE ENTRADAS CON PYDANTIC (CASOS LÍMITE)
# ==============================================================================
def test_api_rechaza_valores_fuera_de_rango_con_422():
    """Caso límite: Pydantic debe rechazar con HTTP 422 horas inválidas o capacidad <= 0."""
    payload_hora_invalida = {**PAYLOAD_VALIDO_P1_P2, "hora": 28}
    resp_hora = cliente_api.post("/predict/p1-demanda", json=payload_hora_invalida)
    assert resp_hora.status_code == 422

    payload_capacidad_cero = {**PAYLOAD_VALIDO_P1_P2, "dpcapacity": 0.0}
    resp_cap = cliente_api.post("/predict/p2-riesgo", json=payload_capacidad_cero)
    assert resp_cap.status_code == 422

    payload_proporcion_invalida = {**PAYLOAD_VALIDO_P3, "prop_fin_de_semana": 1.5}
    resp_p3 = cliente_api.post("/predict/p3-grupos", json=payload_proporcion_invalida)
    assert resp_p3.status_code == 422


def test_api_rechaza_campos_extra_o_datos_personales():
    """Caso límite: Si se envía un campo desconocido o dato personal (Usuario_Id), devuelve 422."""
    payload_con_dato_personal = {**PAYLOAD_VALIDO_P1_P2, "Usuario_Id": 99999}
    respuesta = cliente_api.post("/predict/p1-demanda", json=payload_con_dato_personal)
    assert respuesta.status_code == 422