"""
Escenario de pruebas de carga con Locust para la arquitectura de microservicios MiBici (Equipo 35).

Simula múltiples usuarios concurrentes consultando los endpoints síncronos (P1, P2 y P3),
el flujo asíncrono mediante cola Redis + RQ (/predict/async/p1-demanda y /jobs/{job_id})
y el endpoint de salud (/health).

Ejecución manual (ejemplo en modo headless):
    locust -f load_tests/locustfile.py --headless --host http://127.0.0.1:8000 \
        -u 50 -r 5 -t 60s --csv=reports/operations/locust_pico \
        --html=reports/operations/locust_report_pico.html
"""

import random
import time

from locust import HttpUser, between, task

# ==============================================================================
# 1. CATÁLOGO DE ESTACIONES REALISTAS DEL AMG (PARA VARIAR LAS PETICIONES)
# ==============================================================================
# Coordenadas, capacidades y rangos operativos válidos según mibici/api/schemas.py
ESTACIONES_AMG = [
    {"nombre": "GDL-003_Chapultepec", "dpcapacity": 19.0, "latitude": 20.66769, "longitude": -103.368252},
    {"nombre": "GDL-014_Centro", "dpcapacity": 23.0, "latitude": 20.67667, "longitude": -103.34722},
    {"nombre": "ZAP-010_ZapopanCentro", "dpcapacity": 15.0, "latitude": 20.72094, "longitude": -103.39181},
    {"nombre": "GDL-045_Minerva", "dpcapacity": 27.0, "latitude": 20.67441, "longitude": -103.38739},
]

HORAS_PICO = {7, 8, 9, 17, 18, 19}


# ==============================================================================
# 2. FUNCIONES LINEALES PARA GENERAR PAYLOADS VÁLIDOS DE P1/P2 Y P3
# ==============================================================================
def generar_payload_horario() -> dict:
    """Construye un JSON válido con las 15 variables horarias que esperan P1 y P2."""
    estacion = random.choice(ESTACIONES_AMG)
    hora = random.randint(6, 22)
    dia_semana = random.randint(0, 6)
    es_fin_de_semana = 1 if dia_semana >= 5 else 0
    es_hora_pico = 1 if hora in HORAS_PICO else 0

    # Simulamos mayor volumen de viajes en horas pico
    base_viajes = random.uniform(4.0, 12.0) if es_hora_pico == 1 else random.uniform(0.0, 5.0)
    lag_1_iniciados = round(base_viajes, 2)
    lag_2_iniciados = round(max(0.0, base_viajes + random.uniform(-2.0, 2.0)), 2)
    lag_24_iniciados = round(max(0.0, base_viajes + random.uniform(-1.5, 1.5)), 2)
    media_movil_3 = round((lag_1_iniciados + lag_2_iniciados + lag_24_iniciados) / 3.0, 4)

    lag_1_terminados = round(max(0.0, base_viajes + random.uniform(-4.0, 4.0)), 2)
    lag_1_flujo_neto = round(lag_1_iniciados - lag_1_terminados, 2)
    lag_1_ratio_capacidad = round(abs(lag_1_flujo_neto) / estacion["dpcapacity"], 4)

    return {
        "hora": hora,
        "dia_semana": dia_semana,
        "mes": 10,
        "es_fin_de_semana": es_fin_de_semana,
        "es_hora_pico": es_hora_pico,
        "dpcapacity": estacion["dpcapacity"],
        "latitude": estacion["latitude"],
        "longitude": estacion["longitude"],
        "lag_1_iniciados": lag_1_iniciados,
        "lag_2_iniciados": lag_2_iniciados,
        "lag_24_iniciados": lag_24_iniciados,
        "media_movil_3_iniciados": media_movil_3,
        "lag_1_terminados": lag_1_terminados,
        "lag_1_flujo_neto": lag_1_flujo_neto,
        "lag_1_ratio_capacidad": lag_1_ratio_capacidad,
    }


def generar_payload_perfil_p3() -> dict:
    """Construye un JSON válido con las 9 variables de perfil de estación para P3."""
    estacion = random.choice(ESTACIONES_AMG)
    return {
        "promedio_salidas_diarias": round(random.uniform(12.0, 85.0), 2),
        "duracion_mediana_min": round(random.uniform(7.5, 18.0), 2),
        "prop_fin_de_semana": round(random.uniform(0.12, 0.35), 3),
        "prop_pico_manana": round(random.uniform(0.15, 0.32), 3),
        "prop_pico_tarde": round(random.uniform(0.18, 0.35), 3),
        "ratio_salidas_llegadas": round(random.uniform(0.85, 1.20), 3),
        "dpcapacity": estacion["dpcapacity"],
        "latitude": estacion["latitude"],
        "longitude": estacion["longitude"],
    }


# ==============================================================================
# 3. DEFINICIÓN DEL USUARIO VIRTUAL DE LOCUST
# ==============================================================================
class UsuarioOperativoMiBici(HttpUser):
    """
    Simula un cliente del centro de control de MiBici consultando la API REST
    con una mezcla realista de endpoints síncronos, asíncronos y monitoreo.
    """

    # Pausa corta entre peticiones de un mismo usuario virtual (simula tráfico activo)
    wait_time = between(0.2, 0.8)

    @task(4)
    def predecir_demanda_p1(self):
        """40% del tráfico: predicción síncrona de demanda horaria (P1)."""
        payload = generar_payload_horario()
        with self.client.post(
            "/predict/p1-demanda",
            json=payload,
            name="POST /predict/p1-demanda",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP inesperado en P1: {resp.status_code}")
            elif "viajes_iniciados_predichos" not in resp.json():
                resp.failure("Respuesta P1 sin campo 'viajes_iniciados_predichos'")

    @task(3)
    def predecir_riesgo_p2(self):
        """30% del tráfico: clasificación síncrona de riesgo de desbalance (P2)."""
        payload = generar_payload_horario()
        with self.client.post(
            "/predict/p2-riesgo",
            json=payload,
            name="POST /predict/p2-riesgo",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP inesperado en P2: {resp.status_code}")
            elif "en_riesgo" not in resp.json():
                resp.failure("Respuesta P2 sin campo 'en_riesgo'")

    @task(1)
    def predecir_grupo_p3(self):
        """10% del tráfico: asignación de grupo operativo de estación (P3)."""
        payload = generar_payload_perfil_p3()
        with self.client.post(
            "/predict/p3-grupos",
            json=payload,
            name="POST /predict/p3-grupos",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP inesperado en P3: {resp.status_code}")
            elif "cluster_asignado" not in resp.json():
                resp.failure("Respuesta P3 sin campo 'cluster_asignado'")

    @task(1)
    def predecir_demanda_asincrona_rq(self):
        """10% del tráfico: encola predicción P1 en Redis/RQ y consulta el estado del job."""
        payload = generar_payload_horario()
        job_id = None

        with self.client.post(
            "/predict/async/p1-demanda",
            json=payload,
            name="POST /predict/async/p1-demanda",
            catch_response=True,
        ) as resp:
            if resp.status_code != 202:
                resp.failure(f"HTTP inesperado al encolar en RQ: {resp.status_code}")
                return
            cuerpo = resp.json()
            job_id = cuerpo.get("job_id")
            if not job_id:
                resp.failure("Respuesta asíncrona sin 'job_id'")
                return

        # Pequeña espera y consulta agrupada por plantilla '/jobs/[job_id]'
        time.sleep(0.15)
        with self.client.get(
            f"/jobs/{job_id}",
            name="GET /jobs/[job_id]",
            catch_response=True,
        ) as resp_job:
            if resp_job.status_code != 200:
                resp_job.failure(f"HTTP inesperado al consultar job: {resp_job.status_code}")

    @task(1)
    def verificar_salud_servicio(self):
        """10% del tráfico: verificación periódica de disponibilidad en /health."""
        with self.client.get(
            "/health",
            name="GET /health",
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP inesperado en /health: {resp.status_code}")