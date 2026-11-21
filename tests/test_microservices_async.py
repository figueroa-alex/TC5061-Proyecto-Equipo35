"""
Pruebas automatizadas para la arquitectura de microservicios, cola asíncrona (Redis + RQ),
orquestación con Compose, pruebas de carga y pipeline de CI/CD (Equipo 35).

Verifica sin requerir Redis encendido localmente (usando objetos simulados ligeros):
  1. La tarea del worker RQ (ejecutar_prediccion_p1_asincrona) produce predicciones válidas.
  2. Los endpoints POST /predict/async/p1-demanda y GET /jobs/{job_id} manejan
     encolado (202), consulta finalizada (200), trabajo inexistente (404) y Redis caído (503).
  3. Los archivos docker-compose.yml, load_tests/locustfile.py y
     .github/workflows/ci-cd-mibici.yml existen y contienen la configuración esperada.
"""

from fastapi.testclient import TestClient
from rq.exceptions import NoSuchJobError
import yaml

from mibici import config
from mibici.api import main as api_main
from mibici.api import tasks as api_tasks

cliente_api = TestClient(api_main.app)

PAYLOAD_VALIDO_P1 = {
    "hora": 8,
    "dia_semana": 1,
    "mes": 10,
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


# ==============================================================================
# 1. PRUEBA DIRECTA DE LA TAREA DEL WORKER RQ (mibici/api/tasks.py)
# ==============================================================================
def test_tarea_worker_rq_prediccion_p1_valida():
    """Comprueba que la función que ejecuta rq-worker calcule una demanda >= 0."""
    resultado = api_tasks.ejecutar_prediccion_p1_asincrona(PAYLOAD_VALIDO_P1)
    assert resultado["question"] == "p1"
    assert resultado["model_name"] == config.MODEL_NAME_P1
    assert resultado["model_alias"] == config.PRODUCTION_ALIAS
    assert isinstance(resultado["viajes_iniciados_predichos"], float)
    assert resultado["viajes_iniciados_predichos"] >= 0.0


# ==============================================================================
# 2. PRUEBAS DE ENDPOINTS ASÍNCRONOS (/predict/async/p1-demanda y /jobs/{job_id})
# ==============================================================================
def test_endpoint_async_encola_y_consulta_trabajo_simulado(monkeypatch):
    """Simula Redis/RQ para validar el flujo completo 202 (encolado) -> 200 (finished)."""
    resultado_esperado = api_tasks.ejecutar_prediccion_p1_asincrona(PAYLOAD_VALIDO_P1)

    class TrabajoSimulado:
        id = "job-prueba-equipo35-001"
        is_finished = True
        is_failed = False
        exc_info = None

        def get_status(self, refresh=False):
            return "finished" if refresh else "queued"

        def return_value(self):
            return resultado_esperado

    class ColaSimulada:
        def __init__(self, name, connection):
            self.name = name
            self.connection = connection

        def enqueue(self, func, *args, **kwargs):
            return TrabajoSimulado()

    monkeypatch.setattr(api_main, "obtener_conexion_redis", lambda: object())
    monkeypatch.setattr(api_main, "Queue", ColaSimulada)
    monkeypatch.setattr(api_main.Job, "fetch", lambda job_id, connection: TrabajoSimulado())

    # 1. Encolar predicción asíncrona
    resp_encolado = cliente_api.post("/predict/async/p1-demanda", json=PAYLOAD_VALIDO_P1)
    assert resp_encolado.status_code == 202
    datos_encolado = resp_encolado.json()
    assert datos_encolado["job_id"] == "job-prueba-equipo35-001"
    assert datos_encolado["queue_name"] == config.RQ_QUEUE_NAME
    assert datos_encolado["status"] == "queued"

    # 2. Consultar resultado del trabajo
    resp_job = cliente_api.get("/jobs/job-prueba-equipo35-001")
    assert resp_job.status_code == 200
    datos_job = resp_job.json()
    assert datos_job["status"] == "finished"
    assert datos_job["resultado"]["question"] == "p1"
    assert datos_job["resultado"]["viajes_iniciados_predichos"] >= 0.0


def test_endpoint_jobs_devuelve_404_si_trabajo_no_existe(monkeypatch):
    """Si el job_id no existe en Redis/RQ, GET /jobs/{job_id} debe devolver HTTP 404."""
    def lanzar_no_encontrado(job_id, connection):
        raise NoSuchJobError(f"No existe {job_id}")

    monkeypatch.setattr(api_main, "obtener_conexion_redis", lambda: object())
    monkeypatch.setattr(api_main.Job, "fetch", lanzar_no_encontrado)

    respuesta = cliente_api.get("/jobs/job-inexistente-999")
    assert respuesta.status_code == 404


def test_endpoint_async_devuelve_503_si_redis_no_responde(monkeypatch):
    """Si Redis no está disponible, POST /predict/async/p1-demanda debe devolver HTTP 503."""
    def fallar_conexion():
        raise ConnectionError("Redis no disponible en prueba")

    monkeypatch.setattr(api_main, "obtener_conexion_redis", fallar_conexion)

    respuesta = cliente_api.post("/predict/async/p1-demanda", json=PAYLOAD_VALIDO_P1)
    assert respuesta.status_code == 503


# ==============================================================================
# 3. VALIDACIÓN DE ARCHIVOS DE ORQUESTACIÓN, CARGA Y CI/CD DEL REPOSITORIO
# ==============================================================================
def test_docker_compose_orquesta_cuatro_servicios_con_red_volumenes_y_healthchecks():
    """Comprueba que docker-compose.yml defina los 4 servicios y optimizaciones clave."""
    ruta_compose = config.PROJ_ROOT / "docker-compose.yml"
    assert ruta_compose.exists(), "Falta docker-compose.yml en la raíz del proyecto."

    compose = yaml.safe_load(ruta_compose.read_text(encoding="utf-8"))
    servicios = compose.get("services", {})
    servicios_esperados = {"mlflow-server", "redis", "mibici-api", "rq-worker"}
    assert servicios_esperados.issubset(set(servicios.keys()))

    # Todos los servicios deben tener healthcheck y pertenecer a la red mibici-net
    for nombre in servicios_esperados:
        assert "healthcheck" in servicios[nombre], f"Falta healthcheck en {nombre}"
        assert "mibici-net" in servicios[nombre].get("networks", [])

    # Verificar red dedicada y volúmenes persistentes
    assert "mibici-net" in compose.get("networks", {})
    assert {"mlflow_data", "redis_data"}.issubset(set(compose.get("volumes", {}).keys()))


def test_archivos_locust_finops_y_workflow_cicd_existen():
    """Verifica que existan los scripts de Locust/FinOps y el workflow de GitHub Actions."""
    ruta_locust = config.PROJ_ROOT / "load_tests" / "locustfile.py"
    ruta_finops = config.PROJ_ROOT / "load_tests" / "analizar_carga_finops.py"
    ruta_workflow = config.PROJ_ROOT / ".github" / "workflows" / "ci-cd-mibici.yml"

    assert ruta_locust.exists(), "Falta load_tests/locustfile.py"
    assert ruta_finops.exists(), "Falta load_tests/analizar_carga_finops.py"
    assert ruta_workflow.exists(), "Falta .github/workflows/ci-cd-mibici.yml"

    workflow = yaml.safe_load(ruta_workflow.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs", {})
    assert "pruebas-y-calidad" in jobs
    assert "build-y-despliegue-compose" in jobs
    assert "publicar-imagen-ghcr" in jobs
    assert "notificacion-resumen-pipeline" in jobs