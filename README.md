# Proyecto MiBici: Propuesta y Base Reproducible de Machine Learning (Equipo 35)

Repositorio oficial del **Equipo 35** para la materia **TC5061 — Operaciones de Aprendizaje Automático (MLOps)**. Este proyecto implementa un flujo reproducible de ingesta, limpieza, versionado de datos (SHA-256), ingeniería de características libre de fuga de información (*data leakage*) y seguimiento de experimentos base con **MLFlow Tracking** sobre los datos abiertos del sistema de bicicleta pública **MiBici** del Área Metropolitana de Guadalajara.

---

## 1. Descripción del Proyecto y Preguntas de Negocio

El sistema aborda tres problemas complementarios de aprendizaje automático:

| Código | Pregunta de negocio | Tipo de tarea ML | Unidad de análisis y horizonte | Usuario final y decisión operativa |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | ¿Cuántos viajes se iniciarán en una estación en un periodo futuro? | **Regresión** (`Ridge`) | Estación $\times$ Hora (predicción para la siguiente hora $t$) | **Operador del sistema:** dimensionar la flota activa y anticipar picos horarios de demanda. |
| **P2** | ¿Una estación estará en riesgo de quedarse sin bicicletas o sin anclajes disponibles en un periodo futuro? | **Clasificación binaria** (`LogisticRegression` balanceada) | Estación $\times$ Hora (riesgo $= 1$ si $\Vert{}\text{salidas}_t - \text{llegadas}_t\Vert{} \ge 0.30 \times \text{dpcapacity}$) | **Equipo de rebalanceo:** priorizar el envío de camionetas de redistribución antes de que la estación se vacíe o se sature. |
| **P3** | ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad? | **Clustering** (`K-Means`, $K=4$) | Estación (resumen histórico de comportamiento y coordenadas) | **Planeación urbana y expansión:** diseñar políticas diferenciadas por zona metropolitana y planear nuevas estaciones. |

---

## 2. Datos Utilizados y Mecanismo de Versionado

* **Fuente oficial:** Portal de Datos Abiertos de MiBici (`https://mibici.net/es/datos-abiertos/`).
* **Rango temporal seleccionado:** **Octubre de 2022 a Septiembre de 2026** (`2022-10_a_2026-09`, 48 archivos CSV mensuales de viajes más el catálogo `nomenclatura_2026_09.csv`).
* **Justificación del rango:** El 13 de octubre de 2022 se publicó en el Periódico Oficial *El Estado de Jalisco* el Acuerdo Gubernamental `DIELAG ACU 065/2022` que abrogó las medidas sanitarias obligatorias por COVID-19[cite: 3, 5, 12], normalizando los patrones de movilidad urbana.
* **Volumen y limpieza aplicada (`mibici/dataset.py`):**
  * **Filas crudas leídas:** `17,927,033` registros.
  * **Filas limpias escritas en Parquet:** `17,036,839` registros (se eliminaron `889,481` viajes menores a 121 segundos y `713` duplicados exactos; se estandarizaron `15,859` años de nacimiento fuera de `[1900, 2026]` y `116,260` valores nulos/vacíos de `Genero`).
  * **Estaciones:** `484` estaciones en catálogo (`473` estaciones con viajes activos en el periodo).
* **Mecanismo de versionado (`data/processed/data_version.json`):** Al terminar la ingesta, `mibici/dataset.py` calcula con `hashlib.sha256` el hash criptográfico y el conteo de filas de cada uno de los 48 archivos CSV, del catálogo de nomenclatura y del archivo consolidado `datos_mibici.parquet`. La versión oficial actual de los datos es **`128979137b46`**.

---

## 3. Estructura del Repositorio (Cookiecutter Data Science)

El proyecto fue generado con `cookiecutter-data-science` (`ccds`) activando `include_code_scaffold = Yes`:

```text
TC5061-Proyecto-Equipo35/
├── LICENSE
├── Makefile
├── README.md                        <- Documentación principal y guía de reproducción
├── pyproject.toml                   <- Metadatos del paquete instalable 'mibici'
├── requirements.txt                 <- Dependencias con versiones fijas (==)
├── .gitignore                       <- Excluye CSVs, Parquet, mlruns/, mlflow.db y modelos
│
├── data/
│   ├── raw/                         <- CSVs originales descargados (ignorados en Git)
│   ├── interim/                     <- Datos intermedios (ignorados en Git)
│   └── processed/
│       └── data_version.json        <- Manifiesto versionado en Git con hashes SHA-256 y conteos
│
├── notebooks/
│   └── 1.0-equipo35-exploracion-mibici.ipynb <- Notebook de exploración (convención num-ini-desc)
│
├── references/
│   ├── README_proceso_datos_mibici.txt       <- Documentación detallada de reglas de limpieza
│   ├── ml_canvas_equipo35.md                 <- ML Canvas del proyecto
│   └── arquitectura_equipo35.md              <- Diagrama y justificación de arquitectura
│
├── reports/
│   └── figures/                     <- Gráficas generadas para P1, P2 y P3
│
└── mibici/                          <- Paquete principal de código fuente Python
    ├── __init__.py
    ├── config.py                    <- Configuración centralizada (rutas, fechas, semilla=42)
    ├── dataset.py                   <- Descarga automática, limpieza, Parquet y versionado SHA-256
    ├── features.py                  <- Funciones reutilizables de Feature Engineering (P1, P2 y P3)
    ├── plots.py                     <- Funciones de generación de artefactos visuales
    └── modeling/
        └── train.py                 <- Entrenamiento de baselines/modelos y registro en MLFlow
```

---

## 4. Instrucciones Paso a Paso para Reproducir el Proyecto desde un Clon Limpio
```bash
# 1. Clonar el repositorio y entrar a la carpeta del proyecto
git clone https://github.com/figueroa-alex/TC5061-Proyecto-Equipo35.git

cd TC5061-Proyecto-Equipo35

# 2. Crear y activar el entorno virtual de Python
python3 -m venv .venv

source .venv/bin/activate

# 3. Instalar las dependencias con versión fija y el paquete local 'mibici'
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 4. Ejecutar la ingesta reproducible (verifica o descarga los 48 CSV de Oct 2022 a Sep 2026,
#    limpia por bloques, genera data/processed/datos_mibici.parquet y actualiza data_version.json)
python -m mibici.dataset

# 5. Ejecutar el experimento base de P1, P2 y P3 y registrarlo en MLFlow Tracking
python -m mibici.modeling.train

# 6. Levantar la interfaz web de MLFlow para inspeccionar los runs, métricas y artefactos
mlflow ui --backend-store-uri sqlite:///mlflow.db

#Una vez ejecutado el paso 6, abre en tu navegador http://127.0.0.1:5000 y selecciona el experimento MiBici_Base_Equipo35.
```

---

## 5. Flujo de Trabajo Colaborativo con Git y GitHub (Equipo 35)

El equipo adoptó **GitHub Flow** con las siguientes reglas:

1. **Protección de la rama main:** La rama main está protegida en GitHub y prohíbe *pushes* directos; todo cambio requiere abrir un *Pull Request* (PR) y contar con al menos **1 revisión y aprobación** de otro integrante antes de fusionarse (*merge*).  
2. **Convención de nombres de ramas:** Se utiliza el formato `<tipo>/<descripcion-corta-en-kebab-case>`, por ejemplo:  
   * feat/ingesta-y-configuracion (PR \#1)  
   * feat/ingenieria-caracteristicas (PR \#2)  
   * feat/experimento-base-mlflow (PR \#3)  
   * docs/readme-y-documentacion (PR \#4)  
3. **Convención de mensajes de commit:** Basada en *Conventional Commits* (feat:, fix:, docs:, chore:) seguidos de una descripción concisa en español.  
4. **Exclusión de artefactos pesados:** El archivo .gitignore bloquea cualquier archivo .csv, .parquet, modelos serializados, mlruns/ y mlflow.db para mantener el repositorio ligero y auditable.

---

## 6. Equipo 35

| Nombre | Matrícula | Rol |
|---|---|---|
| Alejandro Figueroa Martínez | A01796930 | Científico de Datos |
| Fernando Gómez Moreno | A00354097 | Ingeniero de Datos |
| Luis Pablo Guerrero Garza | A01840428 | Ingeniero de Machine Learning |
| Rosina Flores Ruiz | A01639950 | Analista de Datos |

