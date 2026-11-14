"""
Archivo de configuración centralizada del proyecto MiBici (Equipo 35).
Aquí se definen todas las rutas de carpetas, el rango de fechas del estudio,
las reglas de limpieza, la semilla aleatoria, los hiperparámetros por defecto
y la configuración de MLflow Tracking y Model Registry, para que ningún otro
archivo tenga valores escritos de forma fija (hardcodeados).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ==============================================================================
# 1. RUTAS PRINCIPALES DEL PROYECTO Y CARGA DE VARIABLES DE ENTORNO (.env)
# ==============================================================================
# Obtenemos la carpeta raíz del repositorio (dos niveles arriba de este archivo)
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Cargamos las variables de entorno desde el archivo .env local si existe.
# El archivo .env está excluido en .gitignore para no exponer configuraciones locales.
load_dotenv(PROJ_ROOT / ".env")

# Carpetas estándar de datos según Cookiecutter Data Science
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Subcarpetas donde se guardan los CSV descargados de MiBici y la nomenclatura
RAW_TRIPS_DIR = RAW_DATA_DIR / "datos_mibici"
RAW_NOMENCLATURE_DIR = RAW_DATA_DIR / "nomenclatura"

# Ruta del archivo de nomenclatura de estaciones (Septiembre 2026)
NOMENCLATURE_FILE = RAW_NOMENCLATURE_DIR / "2026" / "nomenclatura_2026_09.csv"

# Ruta del archivo Parquet consolidado y limpio (salida de dataset.py)
PARQUET_FILE = PROCESSED_DATA_DIR / "datos_mibici.parquet"

# Rutas de tablas intermedias de features (en data/interim/ para acelerar corridas).
# Se construyen la primera vez que se entrena P1/P2 o P3 y después se reutilizan.
INTERIM_P1_P2_FILE = INTERIM_DATA_DIR / "features_p1_p2.parquet"
INTERIM_P3_FILE = INTERIM_DATA_DIR / "features_p3.parquet"

# Ruta del archivo JSON que guarda la versión de los datos (hashes y conteos)
DATA_VERSION_FILE = PROCESSED_DATA_DIR / "data_version.json"

# Carpetas para guardar modelos entrenados y gráficas del reporte
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"

# Carpeta de las gráficas de desempeño. Cada corrida sobrescribe las gráficas de su
# pregunta (p1_*, p2_*, p3_*), así que aquí quedan las de la ÚLTIMA corrida de cada
# pregunta. La copia de cada corrida también se guarda en MLflow, en 'plots/'.
FIGURES_DIR = REPORTS_DIR / "figures"

# ==============================================================================
# 2. PARÁMETROS DE DESCARGA Y RANGO DE FECHAS DEL ESTUDIO
# ==============================================================================
# Página oficial de datos abiertos de MiBici (configurable por variable de entorno)
MIBICI_OPEN_DATA_URL = os.getenv(
    "MIBICI_OPEN_DATA_URL", "https://mibici.net/es/datos-abiertos/"
)

# Esquemas de URL permitidos por seguridad (evita ataques file:// o esquemas personalizados)
ALLOWED_URL_SCHEMES = ("https", "http")

# Tiempo máximo de espera (segundos) por cada descarga antes de marcar error
REQUEST_TIMEOUT_SECONDS = int(os.getenv("MIBICI_REQUEST_TIMEOUT", "60"))

# Rango de fechas seleccionado: Octubre 2022 a Septiembre 2026 (48 meses)
# Justificación: El 13 de octubre de 2022 se publicó el Acuerdo DIELAG ACU 065/2022
# en el Periódico Oficial del Estado de Jalisco, levantando las restricciones
# sanitarias obligatorias por COVID-19 y normalizando la movilidad urbana.
START_YEAR_MONTH = (2022, 10)  # Octubre de 2022
END_YEAR_MONTH = (2026, 9)     # Septiembre de 2026

# Cadena descriptiva del rango de fechas para registrar en MLflow
DATE_RANGE_LABEL = "2022-10_a_2026-09"

# ==============================================================================
# 3. REGLAS DE LIMPIEZA Y VALIDACIÓN DE DATOS
# ==============================================================================
# Nombres estándar de las 8 columnas de los archivos CSV de viajes.
# Usuario_Id, Genero y Año_de_nacimiento son datos personales: se conservan en el
# Parquet limpio, pero NUNCA se usan como variables ni se registran en MLflow.
TRIP_COLUMNS = [
    "Viaje_Id",
    "Usuario_Id",
    "Genero",
    "Año_de_nacimiento",
    "Inicio_del_viaje",
    "Fin_del_viaje",
    "Origen_Id",
    "Destino_Id",
]

# Rango válido para el año de nacimiento de los usuarios.
# Los años fuera de este rango se reemplazan por vacío (no se elimina el viaje).
MIN_BIRTH_YEAR = 1900
MAX_BIRTH_YEAR = 2026

# Duración mínima del viaje en segundos (se eliminan viajes menores a 121 s).
# Retira cancelaciones inmediatas, pruebas técnicas, desbloqueos fallidos o
# bicicletas devueltas por desperfectos (ver references/README_proceso_datos_mibici.txt)
MIN_TRIP_DURATION_SECONDS = 121

# Tamaño de bloque (filas) para leer los CSV sin saturar la memoria RAM
CSV_CHUNK_SIZE = 100000

# ==============================================================================
# 4. PARÁMETROS DE PARTICIÓN TEMPORAL, SEMILLA Y PROBLEMA DE NEGOCIO
# ==============================================================================
# Semilla aleatoria fija para garantizar reproducibilidad en todo el proyecto
RANDOM_SEED = int(os.getenv("MIBICI_RANDOM_SEED", "42"))

# Modo de carga de datos para pruebas automatizadas ('auto', 'real' o 'synthetic')
TEST_DATA_MODE = os.getenv("MIBICI_TEST_DATA_MODE", "auto").lower()

# Fechas de corte para la partición estrictamente temporal (sin usar el futuro):
# - Entrenamiento (Train): Octubre 2022 a Diciembre 2025 (39 meses)
# - Validación (Validation): Enero 2026 a Mayo 2026 (5 meses)
# - Prueba (Test): Junio 2026 a Septiembre 2026 (4 meses)
# Son las mismas fechas de la entrega anterior, para que la comparación contra
# el baseline sea válida.
TRAIN_END_DATE = "2025-12-31 23:59:59"
VAL_END_DATE = "2026-05-31 23:59:59"

# Umbral para P2 (Clasificación): proporción de la capacidad (dpcapacity)
# Si el flujo neto absoluto |salidas - llegadas| en 1 hora es >= 30% de la
# capacidad de la estación, se considera que la estación está en riesgo (1).
RISK_CAPACITY_RATIO_THRESHOLD = 0.30

# ==============================================================================
# 5. VALORES POR DEFECTO DE ENTRENAMIENTO (USADOS POR MLPROJECT Y TRAIN.PY)
# ==============================================================================
# El archivo MLproject envía el texto "from_config" cuando no se indica un
# parámetro con -P; en ese caso train.py toma el valor de esta sección.

# Pregunta y etapa de trabajo por defecto ('baseline', 'search' o 'candidate')
DEFAULT_QUESTION = "p1"
DEFAULT_STAGE = "candidate"

# Modelos por defecto para cada pregunta
DEFAULT_MODEL_P1 = "ridge"
DEFAULT_MODEL_P2 = "logreg"
DEFAULT_MODEL_P3 = "kmeans"

# Hiperparámetros por defecto centralizados
DEFAULT_ALPHA = 1.0             # Regularización de Ridge / Lasso (P1)
DEFAULT_C_REG = 1.0             # Inverso de la regularización de LogisticRegression (P2)
DEFAULT_N_ESTIMATORS = 150      # Árboles (RandomForest) o iteraciones (HistGradientBoosting)
DEFAULT_MAX_DEPTH = 10          # Profundidad máxima de los árboles
DEFAULT_LEARNING_RATE = 0.10    # Tasa de aprendizaje de HistGradientBoosting
DEFAULT_N_CLUSTERS = 4          # Número de grupos de estaciones (P3)

# Submuestreo por árbol en Random Forest para entrenar rápido en millones de filas
RF_MAX_SAMPLES = 0.15

# Grillas de hiperparámetros para las búsquedas con Nested Runs (Paso 4):
# una corrida padre por búsqueda y una corrida hija por cada combinación.
SEARCH_GRID_P1 = [
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10},
    {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.10},
    {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.08},
    {"n_estimators": 250, "max_depth": 12, "learning_rate": 0.08},
]

SEARCH_GRID_P2 = [
    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.10},
    {"n_estimators": 100, "max_depth": 10, "learning_rate": 0.10},
    {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.08},
    {"n_estimators": 250, "max_depth": 12, "learning_rate": 0.08},
]

SEARCH_GRID_P3 = [
    {"n_clusters": 3},
    {"n_clusters": 4},
    {"n_clusters": 5},
    {"n_clusters": 6},
]

# ==============================================================================
# 6. CONFIGURACIÓN DE MLFLOW SERVER, EXPERIMENTO Y MODEL REGISTRY
# ==============================================================================
# Servidor de MLflow Tracking (Paso 1). Se toma de la variable de entorno
# MLFLOW_TRACKING_URI; si no existe, se usa el servidor local en el puerto 5000.
# El servidor guarda todo en una base SQLite (mlflow.db), requisito del Model Registry.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

# Experimento oficial del equipo (configurable mediante variable de entorno)
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mibici-equipo35")

# Etiqueta obligatoria de equipo en todas las corridas
TEAM_TAG = "Equipo35"

# Nombres oficiales de los modelos en el Model Registry (Paso 7)
MODEL_NAME_P1 = "mibici-equipo35-p1-demanda"
MODEL_NAME_P2 = "mibici-equipo35-p2-riesgo"
MODEL_NAME_P3 = "mibici-equipo35-p3-grupos"

# Esquema vigente de ciclo de vida mediante Aliases (Paso 8):
# - champion: versión en producción (se carga con models:/<nombre>@champion)
# - challenger: siguiente mejor versión, lista para reemplazar al champion
PRODUCTION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"

# ==============================================================================
# 7. CONFIGURACIÓN DEL SERVICIO API REST (FASTAPI) Y EMPAQUETADO EN CONTENEDOR
# ==============================================================================
# Carpeta donde se guardan los modelos descargados desde el Model Registry de MLflow
# para empaquetarlos dentro de la imagen del contenedor (estrategia híbrida).
REGISTRY_BUNDLE_DIR = MODELS_DIR / "registry_bundle"
BUNDLE_DIR_P1 = REGISTRY_BUNDLE_DIR / "p1_demanda"
BUNDLE_DIR_P2 = REGISTRY_BUNDLE_DIR / "p2_riesgo"
BUNDLE_DIR_P3 = REGISTRY_BUNDLE_DIR / "p3_grupos"
BUNDLE_MANIFEST_FILE = REGISTRY_BUNDLE_DIR / "registry_manifest.json"

# Alias y versión oficial del Model Registry para recuperar los modelos en la API
MODEL_REGISTRY_ALIAS = os.getenv("MIBICI_MODEL_ALIAS", PRODUCTION_ALIAS)
MODEL_REGISTRY_VERSION = os.getenv("MIBICI_MODEL_VERSION", "2")

# Modo de recuperación del modelo en la API:
#   'hybrid'   -> Intenta cargar desde MLFLOW_TRACKING_URI y, si no está disponible,
#                 usa el modelo descargado del Model Registry en models/registry_bundle/
#   'registry' -> Fuerza la carga en vivo desde el servidor MLFLOW_TRACKING_URI
#   'bundle'   -> Carga directamente el modelo empaquetado en models/registry_bundle/
MODEL_LOAD_MODE = os.getenv("MIBICI_MODEL_LOAD_MODE", "hybrid").lower()

# Host y puerto del servidor FastAPI / Uvicorn
API_HOST = os.getenv("MIBICI_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("MIBICI_API_PORT", "8000"))

# ==============================================================================
# 8. MONITOREO DE DRIFT CON EVIDENTLY, CORTE OCTUBRE 2026 Y A/B TESTING
# ==============================================================================
# Periodo actual de producción para monitoreo de drift y evaluación A/B: Octubre 2026
CURRENT_PERIOD_YEAR_MONTH = (2026, 10)
CURRENT_PERIOD_LABEL = "2026-10"
CURRENT_START_DATE = "2026-10-01 00:00:00"
CURRENT_END_DATE = "2026-10-31 23:59:59"

# Rutas de los archivos crudos del periodo actual (Octubre 2026)
RAW_TRIP_FILE_2026_10 = RAW_TRIPS_DIR / "2026" / "datos_abiertos_2026_10.csv"
NOMENCLATURE_FILE_2026_10 = RAW_NOMENCLATURE_DIR / "2026" / "nomenclatura_2026_10.csv"

# Rutas del Parquet limpio, manifiesto SHA-256 y tablas de features de Octubre 2026
# Se guardan separados para NO alterar datos_mibici.parquet (128979137b46) de entregas 1-4
PARQUET_FILE_2026_10 = PROCESSED_DATA_DIR / "datos_mibici_2026_10.parquet"
DATA_VERSION_FILE_2026_10 = PROCESSED_DATA_DIR / "data_version_2026_10.json"
INTERIM_P1_P2_2026_10_FILE = INTERIM_DATA_DIR / "features_p1_p2_2026_10.parquet"
INTERIM_P3_2026_10_FILE = INTERIM_DATA_DIR / "features_p3_2026_10.parquet"

# Carpeta de reportes de monitoreo (HTML de Evidently y JSON de métricas de drift / A/B test)
MONITORING_REPORTS_DIR = REPORTS_DIR / "monitoring"
EVIDENTLY_HTML_P1_P2 = MONITORING_REPORTS_DIR / "evidently_drift_p1_p2.html"
EVIDENTLY_HTML_P3 = MONITORING_REPORTS_DIR / "evidently_drift_p3.html"
DRIFT_SUMMARY_JSON = MONITORING_REPORTS_DIR / "drift_summary_equipo35.json"
AB_TEST_SUMMARY_JSON = MONITORING_REPORTS_DIR / "ab_testing_results_equipo35.json"

# Tamaño de muestra determinista para reportes interactivos de Evidently en tablas horarias
# (Evita archivos HTML excesivamente pesados manteniendo representatividad estadística)
EVIDENTLY_SAMPLE_SIZE = int(os.getenv("MIBICI_EVIDENTLY_SAMPLE_SIZE", "50000"))
DRIFT_STAT_THRESHOLD = 0.05

# Versiones del Model Registry a comparar en el ejercicio de A/B Testing:
#   - Variante A (Control): v2 (entrenada con Oct 2022 - Dic 2025, 39 meses)
#   - Variante B (Tratamiento): v3 (reentrenada con Oct 2022 - Sep 2026, 48 meses + ajuste)
AB_VARIANT_A_VERSION = "2"
AB_VARIANT_B_VERSION = "3"