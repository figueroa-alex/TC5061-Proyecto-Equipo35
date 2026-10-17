"""
Archivo de configuración centralizada del proyecto MiBici (Equipo 35).
Aquí se definen todas las rutas de carpetas, el rango de fechas del estudio,
las reglas de limpieza y la semilla aleatoria para que ningún otro archivo
tenga valores escritos de forma fija (hardcodeados).
"""

from pathlib import Path

# ==============================================================================
# 1. RUTAS PRINCIPALES DEL PROYECTO
# ==============================================================================
# Obtenemos la carpeta raíz del repositorio (dos niveles arriba de este archivo)
PROJ_ROOT = Path(__file__).resolve().parents[1]

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

# Ruta del archivo JSON que guarda la versión de los datos (hashes y conteos)
DATA_VERSION_FILE = PROCESSED_DATA_DIR / "data_version.json"

# Carpetas para guardar modelos entrenados y gráficas del reporte
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ==============================================================================
# 2. PARÁMETROS DE DESCARGA Y RANGO DE FECHAS DEL ESTUDIO
# ==============================================================================
# Página oficial de datos abiertos de MiBici
MIBICI_OPEN_DATA_URL = "https://mibici.net/es/datos-abiertos/"
REQUEST_TIMEOUT_SECONDS = 60

# Rango de fechas seleccionado: Octubre 2022 a Septiembre 2026 (48 meses)
# Justificación: El 13 de octubre de 2022 se publicó el Acuerdo DIELAG ACU 065/2022
# en el Periódico Oficial del Estado de Jalisco, levantando las restricciones
# sanitarias obligatorias por COVID-19 y normalizando la movilidad urbana.
START_YEAR_MONTH = (2022, 10)  # Octubre de 2022
END_YEAR_MONTH = (2026, 9)     # Septiembre de 2026

# Cadena descriptiva del rango de fechas para registrar en MLFlow
DATE_RANGE_LABEL = "2022-10_a_2026-09"

# ==============================================================================
# 3. REGLAS DE LIMPIEZA Y VALIDACIÓN DE DATOS
# ==============================================================================
# Nombres estándar de las 8 columnas de los archivos CSV de viajes
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

# Rango válido para el año de nacimiento de los usuarios
MIN_BIRTH_YEAR = 1900
MAX_BIRTH_YEAR = 2026

# Duración mínima del viaje en segundos (se eliminan viajes menores a 121 s)
MIN_TRIP_DURATION_SECONDS = 121

# Tamaño de bloque (filas) para leer los CSV sin saturar la memoria RAM
CSV_CHUNK_SIZE = 100000

# ==============================================================================
# 4. PARÁMETROS DE PARTICIÓN TEMPORAL, MODELADO Y MLFLOW
# ==============================================================================
# Semilla aleatoria fija para garantizar reproducibilidad en todo el proyecto
RANDOM_SEED = 42

# Fechas de corte para la partición estrictamente temporal (sin usar el futuro):
# - Entrenamiento (Train): Octubre 2022 a Diciembre 2025 (39 meses)
# - Validación (Validation): Enero 2026 a Mayo 2026 (5 meses)
# - Prueba (Test): Junio 2026 a Septiembre 2026 (4 meses)
TRAIN_END_DATE = "2025-12-31 23:59:59"
VAL_END_DATE = "2026-05-31 23:59:59"

# Umbral para P2 (Clasificación): proporción de la capacidad (dpcapacity)
# Si el flujo neto absoluto |salidas - llegadas| en 1 hora es >= 30% de la
# capacidad de la estación, se considera que la estación está en riesgo (1).
RISK_CAPACITY_RATIO_THRESHOLD = 0.30

# Configuración de MLFlow Tracking local (base de datos SQLite en la raíz)
MLFLOW_TRACKING_URI = f"sqlite:///{PROJ_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "MiBici_Base_Equipo35"
TEAM_TAG = "Equipo35"