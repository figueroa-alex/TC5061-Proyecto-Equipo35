"""
Archivo de configuración centralizada del proyecto MiBici (Equipo 35).
Aquí se definen todas las rutas de carpetas, el rango de fechas del estudio,
las reglas de limpieza, la semilla aleatoria, los hiperparámetros por defecto
y la configuración de MLflow Tracking y Model Registry, para que ningún otro
archivo tenga valores escritos de forma fija (hardcodeados).
"""

import os
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
# Página oficial de datos abiertos de MiBici (de aquí se descargan los CSV)
MIBICI_OPEN_DATA_URL = "https://mibici.net/es/datos-abiertos/"

# Tiempo máximo de espera (segundos) por cada descarga antes de marcar error
REQUEST_TIMEOUT_SECONDS = 60

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
RANDOM_SEED = 42

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

# Mismo experimento de la entrega anterior (baselines y nuevas corridas juntos)
MLFLOW_EXPERIMENT_NAME = "mibici-equipo35"

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
