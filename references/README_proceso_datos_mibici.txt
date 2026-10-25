DOCUMENTACIÓN DEL PROCESO DE INGESTA, LIMPIEZA Y VERSIONADO DE DATOS DE MIBICI
==============================================================================

Este proyecto implementa un flujo reproducible y automatizado para descargar,
organizar, limpiar, consolidar en formato Parquet y versionar los datos abiertos
del sistema de bicicleta pública MiBici dentro de la estructura estándar de
Cookiecutter Data Science (Equipo 35):

1. mibici/config.py   (Configuración centralizada de rutas, fechas y reglas)
2. mibici/dataset.py  (Módulo ejecutable de ingesta, limpieza y versionado)

Todo el proceso se ejecuta desde la raíz del repositorio con un solo comando,
sin pasos manuales:

    python -m mibici.dataset


1. Responsabilidad de los módulos (mibici/config.py y mibici/dataset.py)
------------------------------------------------------------------------

Toda la configuración del proceso vive centralizada en mibici/config.py:
- Rutas de datos crudos (data/raw/datos_mibici/ y data/raw/nomenclatura/).
- Ruta del archivo Parquet limpio (data/processed/datos_mibici.parquet).
- Ruta del manifiesto de versión (data/processed/data_version.json).
- Rango de meses del estudio: Octubre 2022 a Septiembre 2026 (48 meses).
- Constantes de validación (rango de años de nacimiento, duración mínima de
  viaje y tamaño de bloque de lectura).

Por su parte, mibici/dataset.py ejecuta de forma lineal tres fases:
1. Verifica si existen localmente los 48 archivos CSV de viajes del periodo
   (2022_10 a 2026_09) y el catálogo de estaciones (nomenclatura_2026_09.csv).
   Si falta algún archivo, consulta automáticamente el portal oficial de datos
   abiertos (https://mibici.net/es/datos-abiertos/) y lo descarga.
2. Lee los archivos CSV por bloques (chunks), aplica las reglas de limpieza y
   consolida los registros válidos en un único archivo Apache Parquet.
3. Calcula el hash SHA-256 (mediante hashlib) y el número de registros de cada
   archivo CSV individual, del catálogo de nomenclatura y del archivo Parquet
   final, guardando la evidencia auditable en data/processed/data_version.json.


2. Verificación y descarga automatizada de archivos originales
--------------------------------------------------------------

La función asegurar_archivos_csv() en mibici/dataset.py realiza las siguientes
actividades:

1. Genera la lista cronológica de los 48 meses comprendidos entre octubre de
   2022 (START_YEAR_MONTH = (2022, 10)) y septiembre de 2026
   (END_YEAR_MONTH = (2026, 9)).
2. Verifica la existencia local de cada archivo siguiendo la estructura
   organizada por año dentro de data/raw/:

       data/raw/datos_mibici/2022/datos_abiertos_2022_10.csv
       ...
       data/raw/datos_mibici/2026/datos_abiertos_2026_09.csv

3. Verifica la existencia del catálogo maestro de estaciones en:

       data/raw/nomenclatura/2026/nomenclatura_2026_09.csv

4. Si todos los archivos ya existen localmente, no realiza descargas de red.
5. Si falta algún archivo del periodo o el catálogo de nomenclatura, descarga
   el contenido HTML de https://mibici.net/es/datos-abiertos/, identifica los
   enlaces terminados en .csv, filtra únicamente los archivos faltantes del
   rango configurado y los descarga.
6. Cada descarga se escribe primero en un archivo temporal con extensión .part
   y se renombra al archivo definitivo únicamente cuando la transferencia
   finaliza sin errores, evitando dejar archivos corruptos o incompletos.


3. Justificación del periodo de estudio (Octubre 2022 - Septiembre 2026)
------------------------------------------------------------------------

Se seleccionaron los 48 meses entre octubre de 2022 y septiembre de 2026
porque el 13 de octubre de 2022 se publicó en el Periódico Oficial "El Estado
de Jalisco" el Acuerdo Gubernamental DIELAG ACU 065/2022, el cual abrogó el
acuerdo DIELAG ACU 016/2022 y levantó el carácter obligatorio de las medidas
sanitarias por la pandemia de COVID-19. A partir de ese mes la movilidad
urbana retomó sus patrones habituales, evitando introducir sesgos atípicos de
confinamiento en el entrenamiento de los modelos.


4. Exploración, reglas de limpieza y conversión a Parquet
---------------------------------------------------------

La conversión de los 48 archivos CSV a data/processed/datos_mibici.parquet se
realiza por bloques de 100,000 filas (CSV_CHUNK_SIZE = 100000) usando pandas y
pyarrow. Esto permite procesar casi 18 millones de registros sin saturar la
memoria RAM del equipo.

Durante la lectura de cada bloque se aplican las siguientes reglas de calidad:


4.1. Detección automática de codificación
-----------------------------------------

Los archivos CSV publicados por MiBici pueden estar codificados en UTF-8 o en
Windows-1252 (cp1252). La función detectar_codificacion() inspecciona el
primer megabyte de cada archivo:

- Si es válido en UTF-8, utiliza 'utf-8-sig' (con soporte para BOM).
- Si produce error de decodificación, utiliza 'cp1252'.

Esto previene errores al procesar caracteres especiales como la letra 'ñ' en
la columna Año_de_nacimiento.


4.2. Estandarización de columnas por posición
---------------------------------------------

Todos los archivos de viajes se leen omitiendo la primera línea de encabezado
original (skiprows=1) y asignando por posición el esquema fijo de ocho
columnas definido en config.TRIP_COLUMNS:

    Viaje_Id
    Usuario_Id
    Genero
    Año_de_nacimiento
    Inicio_del_viaje
    Fin_del_viaje
    Origen_Id
    Destino_Id


4.3. Homologación de valores faltantes en Genero
------------------------------------------------

En los archivos originales coexisten celdas vacías ("") y el texto literal
"NULL" para representar datos faltantes. El proceso limpia espacios en blanco
y convierte ambas representaciones en valores nulos nativos (pd.NA):

    "NULL"      -> nulo (pd.NA)
    celda vacía -> nulo (pd.NA)

Los valores válidos "M" y "F" se conservan sin cambios.


4.4. Conversión de columnas numéricas a enteros anulables
---------------------------------------------------------

Las siguientes cinco columnas se convierten al tipo entero anulable 'Int64':

    Viaje_Id
    Usuario_Id
    Año_de_nacimiento
    Origen_Id
    Destino_Id

El uso de 'Int64' permite preservar el tipo entero exacto aun cuando existan
valores faltantes (pd.NA) en alguna fila, sin degradar la columna a números
de punto flotante ni a cadenas de texto.


4.5. Validación de Año_de_nacimiento
------------------------------------

Los valores de Año_de_nacimiento se enmascaran como nulo (pd.NA) cuando:

- No son numéricos.
- Son menores que 1900 (MIN_BIRTH_YEAR = 1900).
- Son mayores que 2026 (MAX_BIRTH_YEAR = 2026).

Esta regla impide que años biológicamente imposibles o errores de captura
(como 1, 199, 200 o 1420) contaminen el análisis posterior.


4.6. Validación y normalización de fechas
-----------------------------------------

La función convertir_columna_fecha() reconoce y valida explícitamente los dos
formatos empleados en los archivos históricos de MiBici:

    %Y-%m-%d %H:%M:%S   (ejemplo: 2026-09-01 00:00:59)
    %d/%m/%Y %H:%M      (ejemplo: 01/11/2023 0:01)

Ambos formatos se transforman a objetos temporales estándar (datetime64[ns])
al escribir el archivo Parquet. Si apareciera un formato desconocido o
mezclado dentro de un bloque, el proceso detiene la ejecución explícitamente
para evitar interpretar de forma invertida el día y el mes.


4.7. Eliminación de viajes muy cortos (ruido operativo)
-------------------------------------------------------

Se calcula la duración en segundos de cada viaje (Fin_del_viaje menos
Inicio_del_viaje) y se descartan los registros cuya duración sea menor a 121
segundos:

    MIN_TRIP_DURATION_SECONDS = 121
    condición de descarte: duracion_seg < 121

Este filtro retira cancelaciones inmediatas, pruebas técnicas, desbloqueos
fallidos o bicicletas devueltas en el mismo puerto por desperfectos mecánicos
en los primeros dos minutos.


4.8. Eliminación de duplicados exactos
--------------------------------------

Sobre los registros que superan el filtro de duración se aplica
drop_duplicates() considerando las ocho columnas de la fila, eliminando
registros repetidos exactamente iguales.


4.9. Escritura segura del archivo Parquet
-----------------------------------------

Durante la consolidación, los bloques limpios se escriben inicialmente en el
archivo temporal datos_mibici.parquet.part. Únicamente cuando los 48 archivos
terminan de procesarse sin errores, el archivo temporal reemplaza a
data/processed/datos_mibici.parquet.


5. Versionado de datos con hashlib (SHA-256) y manifiesto JSON
--------------------------------------------------------------

Para garantizar trazabilidad y reproducibilidad en MLOps sin subir gigabytes
de datos crudos a Git, mibici/dataset.py genera el archivo manifiesto:

    data/processed/data_version.json

Este archivo (que sí se versiona en Git) registra mediante la biblioteca
estándar hashlib (algoritmo SHA-256):

- data_version_short: Los primeros 12 caracteres del hash SHA-256 del Parquet
  consolidado (versión actual: 128979137b46), utilizada como parámetro de
  trazabilidad en MLFlow Tracking.
- parquet_sha256: El hash SHA-256 completo del archivo datos_mibici.parquet.
- Conteos exactos de auditoría (filas crudas leídas, filas limpias escritas,
  años inválidos enmascarados, nulos homologados en Genero, viajes cortos
  eliminados, duplicados eliminados, estaciones en catálogo y estaciones
  activas).
- nomenclature_file: Nombre, número de filas y hash SHA-256 de
  nomenclatura_2026_09.csv.
- source_csv_files: Lista detallada de los 48 archivos CSV procesados con su
  nombre, número de filas y hash SHA-256 individual.


6. Resultados obtenidos en la versión oficial (128979137b46)
------------------------------------------------------------

Al ejecutar el proceso sobre el rango Octubre 2022 - Septiembre 2026 se
obtuvieron las siguientes cifras:

- Periodo procesado:                   2022-10_a_2026-09 (48 archivos CSV)
- Filas crudas leídas:                 17,927,033
- Filas limpias escritas en Parquet:   17,036,839
- Años de nacimiento inválidos -> NA:  15,859
- Género 'NULL' -> NA:                 114,163
- Género vacío -> NA:                  2,097
- Viajes cortos (< 121 s) eliminados:  889,481
- Duplicados exactos eliminados:       713
- Estaciones en catálogo nomenclatura: 484
- Estaciones activas en los viajes:    473
- Versión de datos (SHA-256 corto):    128979137b46


7. Relación con el módulo de Feature Engineering y Modelado
-----------------------------------------------------------

El archivo data/processed/datos_mibici.parquet y el catálogo de nomenclatura
alimentan directamente a mibici/features.py y mibici/modeling/train.py para
resolver las tres preguntas del proyecto:

- P1 (Regresión): Predicción del número de viajes iniciados por estación y
  por hora.
- P2 (Clasificación): Identificación del riesgo horario de quedarse sin
  bicicletas o sin anclajes cuando el desbalance absoluto supera o iguala el
  30% de la capacidad (dpcapacity) de la estación.
- P3 (Clustering): Agrupamiento de estaciones con patrones de uso y ubicación
  similares utilizando exclusivamente el conjunto de entrenamiento para evitar
  fuga de información temporal.