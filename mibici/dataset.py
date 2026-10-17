"""
Script de ingesta, limpieza, conversión a Parquet y versionado de datos de MiBici.
"""

import hashlib
import html
import json
import re
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mibici import config


# ==============================================================================
# FUNCIÓN 1: CALCULAR HASH SHA-256 Y CONTEO DE FILAS DE UN ARCHIVO
# ==============================================================================
def calcular_sha256_archivo(ruta_archivo: Path) -> str:
    """
    Lee un archivo en bloques de 1 MB y devuelve su hash SHA-256 en texto
    hexadecimal usando la librería estándar hashlib.
    """
    calculador = hashlib.sha256()
    with open(ruta_archivo, "rb") as archivo_bin:
        while True:
            bloque = archivo_bin.read(1024 * 1024)
            if not bloque:
                break
            calculador.update(bloque)
    return calculador.hexdigest()


def contar_registros_csv(ruta_csv: Path) -> int:
    """
    Cuenta el número de registros de datos de un archivo CSV (restando 1 por
    la fila de encabezado) de forma rápida sin cargar todo en memoria.
    """
    with open(ruta_csv, "rb") as archivo_bin:
        total_lineas = sum(1 for _ in archivo_bin)
    return max(0, total_lineas - 1)


# ==============================================================================
# FUNCIÓN 2: OBTENER O VERIFICAR ARCHIVOS CSV DEL RANGO OCT 2022 - SEP 2026
# ==============================================================================
def asegurar_archivos_csv() -> tuple[list[Path], Path]:
    """
    Revisa si los archivos CSV de viajes (entre START_YEAR_MONTH y END_YEAR_MONTH)
    y el archivo de nomenclatura ya existen localmente en data/raw/.
    Si falta alguno, lo descarga automáticamente del portal de MiBici.
    """
    config.RAW_TRIPS_DIR.mkdir(parents=True, exist_ok=True)
    config.RAW_NOMENCLATURE_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    patron_nombre = re.compile(
        r"^(?P<tipo>datos_abiertos|nomenclatura)_(?P<anio>\d{4})_(?P<mes>\d{2})[^.]*\.csv$",
        re.IGNORECASE,
    )

    # Generamos la lista de los 48 meses esperados entre (2022, 10) y (2026, 09)
    meses_esperados = []
    anio_actual, mes_actual = config.START_YEAR_MONTH
    anio_fin, mes_fin = config.END_YEAR_MONTH

    while (anio_actual, mes_actual) <= (anio_fin, mes_fin):
        meses_esperados.append((anio_actual, mes_actual))
        if mes_actual == 12:
            anio_actual += 1
            mes_actual = 1
        else:
            mes_actual += 1

    archivos_faltantes = []
    rutas_viajes_locales = []

    for anio, mes in meses_esperados:
        nombre_esperado = f"datos_abiertos_{anio:04d}_{mes:02d}.csv"
        ruta_esperada = config.RAW_TRIPS_DIR / f"{anio:04d}" / nombre_esperado
        if ruta_esperada.exists():
            rutas_viajes_locales.append(ruta_esperada)
        else:
            archivos_faltantes.append((anio, mes))

    ruta_nomenclatura = config.NOMENCLATURE_FILE
    ruta_alternativa_nom = config.RAW_DATA_DIR / "nomenclatura_2026_09.csv"
    if not ruta_nomenclatura.exists() and ruta_alternativa_nom.exists():
        ruta_nomenclatura.parent.mkdir(parents=True, exist_ok=True)
        ruta_alternativa_nom.replace(ruta_nomenclatura)

    if len(archivos_faltantes) > 0 or not ruta_nomenclatura.exists():
        print(
            f"Faltan {len(archivos_faltantes)} archivos CSV de viajes o la nomenclatura. "
            f"Consultando {config.MIBICI_OPEN_DATA_URL}..."
        )
        peticion = Request(
            config.MIBICI_OPEN_DATA_URL,
            headers={"User-Agent": "mibici-csv-downloader/1.0"},
        )
        with urlopen(peticion, timeout=config.REQUEST_TIMEOUT_SECONDS) as respuesta:
            codificacion_web = respuesta.headers.get_content_charset() or "utf-8"
            html_pagina = respuesta.read().decode(codificacion_web, errors="replace")

        enlaces_encontrados = re.findall(
            r'href=["\']([^"\']+\.csv)["\']', html_pagina, re.IGNORECASE
        )
        urls_unicas = []
        for enlace in enlaces_encontrados:
            url_completa = urljoin(
                config.MIBICI_OPEN_DATA_URL, html.unescape(enlace.strip())
            )
            if url_completa not in urls_unicas:
                urls_unicas.append(url_completa)

        for url in urls_unicas:
            nombre_en_url = Path(unquote(urlparse(url).path)).name
            coincidencia = patron_nombre.fullmatch(nombre_en_url)
            if coincidencia is None:
                continue

            tipo = coincidencia.group("tipo").lower()
            anio = int(coincidencia.group("anio"))
            mes = int(coincidencia.group("mes"))
            nombre_limpio = f"{tipo}_{anio:04d}_{mes:02d}.csv"

            if tipo == "datos_abiertos":
                if (anio, mes) not in archivos_faltantes:
                    continue
                destino = config.RAW_TRIPS_DIR / f"{anio:04d}" / nombre_limpio
            else:
                if ruta_nomenclatura.exists():
                    continue
                destino = config.RAW_NOMENCLATURE_DIR / f"{anio:04d}" / nombre_limpio

            destino.parent.mkdir(parents=True, exist_ok=True)
            ruta_temporal = destino.with_name(destino.name + ".part")
            print(f"Descargando: {nombre_limpio} -> {destino}")
            peticion_archivo = Request(
                url, headers={"User-Agent": "mibici-csv-downloader/1.0"}
            )
            with urlopen(
                peticion_archivo, timeout=config.REQUEST_TIMEOUT_SECONDS
            ) as resp_csv, open(ruta_temporal, "wb") as archivo_salida:
                while True:
                    bloque = resp_csv.read(1024 * 1024)
                    if not bloque:
                        break
                    archivo_salida.write(bloque)
            ruta_temporal.replace(destino)

            if tipo == "datos_abiertos":
                rutas_viajes_locales.append(destino)
            elif tipo == "nomenclatura":
                ruta_nomenclatura = destino

    rutas_viajes_locales = sorted(rutas_viajes_locales)
    return rutas_viajes_locales, ruta_nomenclatura


# ==============================================================================
# FUNCIÓN 3: DETECTAR CODIFICACIÓN Y NORMALIZAR FECHAS DE UN BLOQUE (CHUNK)
# ==============================================================================
def detectar_codificacion(ruta_csv: Path) -> str:
    """
    Lee el primer megabyte del archivo CSV para saber si está en UTF-8 o en
    Windows-1252 (cp1252). Así evitamos errores con la letra 'ñ'.
    """
    muestra = ruta_csv.read_bytes()[: 1024 * 1024]
    try:
        muestra.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp1252"


def convertir_columna_fecha(
    serie_texto: pd.Series, nombre_columna: str, ruta_csv: Path
) -> pd.Series:
    """
    Convierte una columna de texto a fecha/hora (datetime), reconociendo los dos
    formatos usados por MiBici: 'YYYY-MM-DD HH:MM:SS' y 'DD/MM/YYYY HH:MM'.
    """
    valores_no_vacios = serie_texto.dropna().astype("string").str.strip()
    valores_no_vacios = valores_no_vacios[valores_no_vacios != ""]
    if valores_no_vacios.empty:
        return pd.to_datetime(serie_texto, errors="coerce")

    patron_iso = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    patron_dia_primero = re.compile(r"\d{2}/\d{2}/\d{4} \d{1,2}:\d{2}")

    if valores_no_vacios.str.fullmatch(patron_iso).all():
        return pd.to_datetime(serie_texto, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    elif valores_no_vacios.str.fullmatch(patron_dia_primero).all():
        return pd.to_datetime(serie_texto, format="%d/%m/%Y %H:%M", errors="coerce")
    else:
        ejemplo = valores_no_vacios.iloc[0]
        raise ValueError(
            f"Formato de fecha no reconocido en {ruta_csv.name} ({nombre_columna}): {ejemplo}"
        )


# ==============================================================================
# FUNCIÓN 4: LIMPIAR CADA ARCHIVO CSV Y GUARDAR EN UN PARQUET CONSOLIDADO
# ==============================================================================
def construir_parquet_y_versionar(
    rutas_csv: list[Path], ruta_nomenclatura: Path
) -> dict:
    """
    Si el archivo Parquet aún no existe, procesa los 48 archivos CSV por bloques,
    aplica las reglas de limpieza y crea datos_mibici.parquet.
    Después, genera el manifiesto data_version.json incluyendo el hash SHA-256
    y el número de registros de cada archivo individual y del Parquet final.
    """
    # Si ya tenemos el Parquet y un manifiesto previo con los conteos de limpieza,
    # reutilizamos los conteos de limpieza para no tardar minutos reconstruyendo el Parquet.
    reconstruir_parquet = not config.PARQUET_FILE.exists()
    datos_previos = {}
    if config.DATA_VERSION_FILE.exists():
        with open(config.DATA_VERSION_FILE, "r", encoding="utf-8") as f_prev:
            datos_previos = json.load(f_prev)

    if reconstruir_parquet or "rows_written_clean" not in datos_previos:
        total_filas_leidas = 0
        total_filas_escritas = 0
        total_anios_invalidos = 0
        total_genero_null = 0
        total_genero_vacio = 0
        total_viajes_cortos = 0
        total_duplicados = 0
        estaciones_origen_activas = set()
        estaciones_destino_activas = set()

        ruta_parquet_temp = config.PARQUET_FILE.with_name(
            config.PARQUET_FILE.name + ".part"
        )
        ruta_parquet_temp.unlink(missing_ok=True)
        escritor_parquet = None
        esquema_parquet = None

        print(
            f"\nIniciando procesamiento de {len(rutas_csv)} archivos CSV (Oct 2022 - Sep 2026)..."
        )
        try:
            for ruta_csv in rutas_csv:
                print(f"  -> Procesando: {ruta_csv.name}")
                codificacion = detectar_codificacion(ruta_csv)
                lector_bloques = pd.read_csv(
                    ruta_csv,
                    encoding=codificacion,
                    header=None,
                    names=config.TRIP_COLUMNS,
                    skiprows=1,
                    chunksize=config.CSV_CHUNK_SIZE,
                    dtype="string",
                    keep_default_na=False,
                    na_filter=False,
                    low_memory=False,
                )

                for bloque in lector_bloques:
                    total_filas_leidas += len(bloque)
                    anio_original_txt = (
                        bloque["Año_de_nacimiento"].astype("string").str.strip()
                    )
                    columnas_enteras = [
                        "Viaje_Id",
                        "Usuario_Id",
                        "Año_de_nacimiento",
                        "Origen_Id",
                        "Destino_Id",
                    ]
                    for col in columnas_enteras:
                        bloque[col] = pd.to_numeric(
                            bloque[col], errors="coerce"
                        ).astype("Int64")

                    anio_num = bloque["Año_de_nacimiento"]
                    tenia_dato_anio = ~anio_original_txt.isin(["", "NULL"])
                    es_anio_invalido = tenia_dato_anio & (
                        anio_num.isna()
                        | (anio_num < config.MIN_BIRTH_YEAR)
                        | (anio_num > config.MAX_BIRTH_YEAR)
                    )
                    total_anios_invalidos += int(es_anio_invalido.sum())
                    bloque["Año_de_nacimiento"] = anio_num.mask(
                        (anio_num < config.MIN_BIRTH_YEAR)
                        | (anio_num > config.MAX_BIRTH_YEAR)
                    )

                    genero_limpio = bloque["Genero"].astype("string").str.strip()
                    total_genero_null += int((genero_limpio == "NULL").sum())
                    total_genero_vacio += int((genero_limpio == "").sum())
                    bloque["Genero"] = genero_limpio.replace({"NULL": pd.NA, "": pd.NA})

                    bloque["Inicio_del_viaje"] = convertir_columna_fecha(
                        bloque["Inicio_del_viaje"], "Inicio_del_viaje", ruta_csv
                    )
                    bloque["Fin_del_viaje"] = convertir_columna_fecha(
                        bloque["Fin_del_viaje"], "Fin_del_viaje", ruta_csv
                    )

                    duracion_seg = (
                        bloque["Fin_del_viaje"] - bloque["Inicio_del_viaje"]
                    ).dt.total_seconds()
                    es_viaje_corto = duracion_seg < config.MIN_TRIP_DURATION_SECONDS
                    total_viajes_cortos += int(es_viaje_corto.sum())

                    bloque_filtrado = bloque.loc[~es_viaje_corto, config.TRIP_COLUMNS]
                    bloque_sin_duplicados = bloque_filtrado.drop_duplicates()
                    total_duplicados += len(bloque_filtrado) - len(
                        bloque_sin_duplicados
                    )
                    total_filas_escritas += len(bloque_sin_duplicados)

                    estaciones_origen_activas.update(
                        bloque_sin_duplicados["Origen_Id"].dropna().astype(int).tolist()
                    )
                    estaciones_destino_activas.update(
                        bloque_sin_duplicados["Destino_Id"]
                        .dropna()
                        .astype(int)
                        .tolist()
                    )

                    tabla_arrow = pa.Table.from_pandas(
                        bloque_sin_duplicados, preserve_index=False
                    )
                    if esquema_parquet is None:
                        esquema_parquet = tabla_arrow.schema
                        escritor_parquet = pq.ParquetWriter(
                            ruta_parquet_temp, esquema_parquet
                        )
                    escritor_parquet.write_table(tabla_arrow.cast(esquema_parquet))

            if escritor_parquet is not None:
                escritor_parquet.close()
                escritor_parquet = None
            ruta_parquet_temp.replace(config.PARQUET_FILE)
        except Exception:
            if escritor_parquet is not None:
                escritor_parquet.close()
            ruta_parquet_temp.unlink(missing_ok=True)
            raise

        estaciones_activas_totales = len(
            estaciones_origen_activas.union(estaciones_destino_activas)
        )
    else:
        print(
            f"\nEl archivo Parquet ya existe en {config.PARQUET_FILE}. "
            "Verificando hashes SHA-256 individuales por archivo..."
        )
        total_filas_leidas = datos_previos["rows_read_raw"]
        total_filas_escritas = datos_previos["rows_written_clean"]
        total_anios_invalidos = datos_previos["invalid_birth_years_masked"]
        total_genero_null = datos_previos["genero_null_to_na"]
        total_genero_vacio = datos_previos["genero_empty_to_na"]
        total_viajes_cortos = datos_previos["short_trips_removed_lt_121s"]
        total_duplicados = datos_previos["exact_duplicates_removed"]
        estaciones_activas_totales = datos_previos["active_stations_in_trips"]

    # ==========================================================================
    # VERSIONADO CON HASHLIB: HASH Y FILAS DE CADA ARCHIVO Y DEL PARQUET FINAL
    # ==========================================================================
    parquet_sha256 = calcular_sha256_archivo(config.PARQUET_FILE)

    df_nomenclatura = pd.read_csv(
        ruta_nomenclatura, encoding=detectar_codificacion(ruta_nomenclatura)
    )
    total_estaciones_catalogo = int(df_nomenclatura["id"].nunique())

    # Calculamos el hash SHA-256 y el número de registros de cada archivo fuente
    detalle_archivos = []
    for ruta_csv in rutas_csv:
        detalle_archivos.append(
            {
                "file": ruta_csv.name,
                "rows": contar_registros_csv(ruta_csv),
                "sha256": calcular_sha256_archivo(ruta_csv),
            }
        )

    manifiesto_version = {
        "data_version_short": parquet_sha256[:12],
        "parquet_sha256": parquet_sha256,
        "date_range": config.DATE_RANGE_LABEL,
        "months_processed": len(rutas_csv),
        "first_file": rutas_csv[0].name if rutas_csv else None,
        "last_file": rutas_csv[-1].name if rutas_csv else None,
        "rows_read_raw": total_filas_leidas,
        "rows_written_clean": total_filas_escritas,
        "invalid_birth_years_masked": total_anios_invalidos,
        "genero_null_to_na": total_genero_null,
        "genero_empty_to_na": total_genero_vacio,
        "short_trips_removed_lt_121s": total_viajes_cortos,
        "exact_duplicates_removed": total_duplicados,
        "stations_in_nomenclature_catalog": total_estaciones_catalogo,
        "active_stations_in_trips": estaciones_activas_totales,
        "nomenclature_file": {
            "file": ruta_nomenclatura.name,
            "rows": len(df_nomenclatura),
            "sha256": calcular_sha256_archivo(ruta_nomenclatura),
        },
        "source_csv_files": detalle_archivos,
    }

    with open(config.DATA_VERSION_FILE, "w", encoding="utf-8") as archivo_json:
        json.dump(manifiesto_version, archivo_json, indent=4, ensure_ascii=False)

    return manifiesto_version


def main() -> None:
    rutas_csv, ruta_nomenclatura = asegurar_archivos_csv()
    if len(rutas_csv) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV entre {config.START_YEAR_MONTH} y {config.END_YEAR_MONTH}."
        )

    reporte = construir_parquet_y_versionar(rutas_csv, ruta_nomenclatura)

    print("\n" + "=" * 60)
    print("RESUMEN DE INGESTA Y VERSIONADO DE DATOS (EQUIPO 35)")
    print("=" * 60)
    print(f"  Periodo procesado:                   {reporte['date_range']}")
    print(f"  Total de meses (archivos CSV):       {reporte['months_processed']}")
    print(
        f"  Archivo inicial y final:             {reporte['first_file']} -> {reporte['last_file']}"
    )
    print(f"  Filas crudas leídas:                 {reporte['rows_read_raw']:,}")
    print(f"  Filas limpias escritas en Parquet:   {reporte['rows_written_clean']:,}")
    print(
        f"  Archivos individuales con SHA-256:   {len(reporte['source_csv_files'])} CSVs + 1 Nomenclatura"
    )
    print(f"  Versión de datos (SHA-256 corto):    {reporte['data_version_short']}")
    print("=" * 60)


if __name__ == "__main__":
    main()