"""
Pruebas unitarias con PyTest para todas las funciones clave del proyecto MiBici (Equipo 35).
Cubre procesamiento de datos (dataset.py), ingeniería de características (features.py),
visualizaciones (plots.py) y utilidades de entrenamiento/configuración (config.py y train.py),
evaluando tanto casos normales como casos límite (edge cases).
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mibici import config, dataset, features, plots
from mibici.modeling import train


# ==============================================================================
# BLOQUE 1: PRUEBAS UNITARIAS DE PROCESAMIENTO Y LIMPIEZA DE DATOS (dataset.py)
# ==============================================================================
def test_validar_url_segura_caso_normal():
    """Caso normal: URLs con esquema https:// o http:// y dominio válido son aceptadas."""
    url_https = "https://mibici.net/es/datos-abiertos/datos_abiertos_2026_09.csv"
    url_http = "http://127.0.0.1:5000"
    assert dataset.validar_url_segura(url_https) == url_https
    assert dataset.validar_url_segura(url_http) == url_http


@pytest.mark.parametrize(
    "url_insegura",
    [
        "file:///etc/passwd",
        "ftp://servidor.interno/datos.csv",
        "javascript:alert(1)",
        "",
        "   ",
        "https:///sin-dominio",
    ],
)
def test_validar_url_segura_casos_limite(url_insegura: str):
    """Caso límite: URLs con esquemas riesgosos (file://, ftp://) o vacías lanzan ValueError."""
    with pytest.raises(ValueError):
        dataset.validar_url_segura(url_insegura)


def test_calcular_sha256_y_contar_registros_csv(tmp_path: Path):
    """
    Caso normal y límite: Verifica que calcular_sha256_archivo coincida con hashlib
    y que contar_registros_csv cuente correctamente filas en archivos con datos y vacíos.
    """
    contenido_csv = "Viaje_Id,Origen_Id\n1,10\n2,20\n3,30\n"
    archivo_prueba = tmp_path / "prueba.csv"
    archivo_prueba.write_text(contenido_csv, encoding="utf-8")

    hash_esperado = hashlib.sha256(contenido_csv.encode("utf-8")).hexdigest()
    assert dataset.calcular_sha256_archivo(archivo_prueba) == hash_esperado
    assert dataset.contar_registros_csv(archivo_prueba) == 3

    # Caso límite: archivo completamente vacío (0 líneas -> debe devolver 0, nunca -1)
    archivo_vacio = tmp_path / "vacio.csv"
    archivo_vacio.write_text("", encoding="utf-8")
    assert dataset.contar_registros_csv(archivo_vacio) == 0


def test_detectar_codificacion_utf8_y_cp1252(tmp_path: Path):
    """Caso normal y límite: Detecta correctamente archivos en UTF-8 y en Windows-1252 (letra ñ)."""
    ruta_utf8 = tmp_path / "utf8.csv"
    ruta_utf8.write_bytes("Año_de_nacimiento,Genero\n1990,M\n".encode("utf-8-sig"))
    assert dataset.detectar_codificacion(ruta_utf8) == "utf-8-sig"

    ruta_cp1252 = tmp_path / "cp1252.csv"
    ruta_cp1252.write_bytes("Año_de_nacimiento,Genero\n1990,F\n".encode("cp1252"))
    assert dataset.detectar_codificacion(ruta_cp1252) == "cp1252"


def test_convertir_columna_fecha_normal_y_casos_limite():
    """
    Prueba convertir_columna_fecha en:
      - Caso normal 1: formato ISO ('YYYY-MM-DD HH:MM:SS').
      - Caso normal 2: formato día primero ('DD/MM/YYYY HH:MM').
      - Caso límite 1: serie completamente vacía.
      - Caso límite 2: formato de fecha inválido o desconocido (debe lanzar ValueError).
    """
    ruta_ficticia = Path("datos_abiertos_2026_09.csv")

    serie_iso = pd.Series(["2026-09-01 08:15:00", "2026-09-01 09:30:00"])
    res_iso = dataset.convertir_columna_fecha(serie_iso, "Inicio_del_viaje", ruta_ficticia)
    assert pd.api.types.is_datetime64_any_dtype(res_iso)
    assert res_iso.iloc[0] == pd.Timestamp("2026-09-01 08:15:00")

    serie_dmy = pd.Series(["15/11/2023 07:05", "15/11/2023 18:45"])
    res_dmy = dataset.convertir_columna_fecha(serie_dmy, "Inicio_del_viaje", ruta_ficticia)
    assert res_dmy.iloc[0] == pd.Timestamp("2023-11-15 07:05:00")

    # Caso límite: serie vacía o solo con espacios
    serie_vacia = pd.Series(["", "   ", None])
    res_vacia = dataset.convertir_columna_fecha(serie_vacia, "Inicio_del_viaje", ruta_ficticia)
    assert res_vacia.isna().all()

    # Caso límite: texto que no es fecha válida
    serie_invalida = pd.Series(["fecha_corrupta_2026"])
    with pytest.raises(ValueError, match="Formato de fecha no reconocido"):
        dataset.convertir_columna_fecha(serie_invalida, "Inicio_del_viaje", ruta_ficticia)


def test_limpiar_bloque_viajes_normal_y_casos_limite():
    """
    Verifica todas las reglas de negocio de limpiar_bloque_viajes:
      - Conserva viajes válidos (>= 121 s).
      - Elimina viajes cortos (< 121 s, por ejemplo 60 s).
      - Elimina duplicados exactos.
      - Enmascara como pd.NA años fuera de [1900, 2026] (ej. 1850 y 2099) sin borrar el viaje.
      - Homologa 'NULL' y '' en Genero a pd.NA.
      - Maneja correctamente un DataFrame vacío (caso límite).
    """
    bloque_crudo = pd.DataFrame(
        {
            "Viaje_Id": ["1", "1", "2", "3"],
            "Usuario_Id": ["101", "101", "102", "103"],
            "Genero": ["M", "M", "NULL", ""],
            "Año_de_nacimiento": ["1992", "1992", "1850", "2099"],
            "Inicio_del_viaje": [
                "2026-06-01 08:00:00",
                "2026-06-01 08:00:00",  # Duplicado exacto de la fila 1
                "2026-06-01 08:10:00",  # Viaje válido (10 min = 600 s) con año inválido 1850
                "2026-06-01 09:00:00",  # Viaje corto (60 s < 121 s) -> debe eliminarse
            ],
            "Fin_del_viaje": [
                "2026-06-01 08:15:00",
                "2026-06-01 08:15:00",
                "2026-06-01 08:20:00",
                "2026-06-01 09:01:00",
            ],
            "Origen_Id": ["10", "10", "20", "30"],
            "Destino_Id": ["20", "20", "30", "10"],
        }
    )

    df_limpio, stats = dataset.limpiar_bloque_viajes(bloque_crudo)

    assert stats["filas_leidas"] == 4
    assert stats["viajes_cortos"] == 1
    assert stats["duplicados"] == 1
    assert stats["anios_invalidos"] == 2
    assert stats["genero_null"] == 1
    assert stats["genero_vacio"] == 1
    assert stats["filas_escritas"] == 2
    assert len(df_limpio) == 2

    # El viaje 2 se conserva, pero su año 1850 y su género 'NULL' quedan como pd.NA
    fila_viaje_2 = df_limpio.loc[df_limpio["Viaje_Id"] == 2].iloc[0]
    assert pd.isna(fila_viaje_2["Año_de_nacimiento"])
    assert pd.isna(fila_viaje_2["Genero"])

    # Caso límite: DataFrame vacío
    df_vacio, stats_vacio = dataset.limpiar_bloque_viajes(
        pd.DataFrame(columns=config.TRIP_COLUMNS)
    )
    assert df_vacio.empty
    assert stats_vacio["filas_escritas"] == 0


# ==============================================================================
# BLOQUE 2: PRUEBAS UNITARIAS DE INGENIERÍA DE CARACTERÍSTICAS (features.py)
# ==============================================================================
def test_construir_tabla_horaria_y_features_p1_p2_sin_fuga(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Caso normal: Verifica que construir_tabla_horaria_estaciones y generar_features_p1_y_p2
    creen todas las columnas esperadas para P1 y P2 y que lag_1_iniciados corresponda
    exactamente al valor anterior (.shift(1)) dentro de cada estación (sin data leakage).
    """
    monkeypatch.setattr(features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica)

    tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes_validos_sinteticos)
    assert not tabla_horaria.empty
    assert {"Estacion_Id", "Fecha_Hora", "viajes_iniciados", "viajes_terminados", "dpcapacity"}.issubset(
        tabla_horaria.columns
    )

    df_features = features.generar_features_p1_y_p2(tabla_horaria)
    assert not df_features.empty
    assert df_features["en_riesgo"].isin([0, 1]).all()

    # Verificación matemática contra fuga de datos (data leakage):
    # En la tabla horaria original de la estación 10, el valor en la fila t-1 debe coincidir
    # con lag_1_iniciados en la fila t.
    est_10_horaria = tabla_horaria.loc[tabla_horaria["Estacion_Id"] == 10].reset_index(drop=True)
    est_10_feat = df_features.loc[df_features["Estacion_Id"] == 10].reset_index(drop=True)
    primera_hora_feat = est_10_feat.loc[0, "Fecha_Hora"]
    idx_en_horaria = est_10_horaria.index[est_10_horaria["Fecha_Hora"] == primera_hora_feat][0]

    valor_real_t_menos_1 = est_10_horaria.loc[idx_en_horaria - 1, "viajes_iniciados"]
    assert est_10_feat.loc[0, "lag_1_iniciados"] == valor_real_t_menos_1


def test_generar_features_p1_p2_caso_limite_capacidad_cero(monkeypatch, df_viajes_validos_sinteticos):
    """
    Caso límite: Si una estación en el catálogo tiene capacidad 0 (dpcapacity = 0),
    generar_features_p1_y_p2 no debe fallar por división entre cero (ZeroDivisionError),
    sino reemplazar 0 por NaN y descartar esas filas de forma segura.
    """
    catalogo_capacidad_cero = pd.DataFrame(
        {
            "Estacion_Id": pd.Series([10, 20], dtype="int64"),
            "name": ["Estacion Capacidad Cero", "Estacion Normal"],
            "latitude": [20.67, 20.68],
            "longitude": [-103.36, -103.37],
            "dpcapacity": pd.Series([0, 20], dtype="int64"),
        }
    )
    monkeypatch.setattr(features, "cargar_nomenclatura", lambda: catalogo_capacidad_cero)

    tabla_horaria = features.construir_tabla_horaria_estaciones(df_viajes_validos_sinteticos)
    df_features = features.generar_features_p1_y_p2(tabla_horaria)

    # La estación 10 (capacidad 0) se filtra sin producir infinitos ni errores
    assert 10 not in df_features["Estacion_Id"].values
    assert (df_features["Estacion_Id"] == 20).all()
    assert not np.isinf(df_features["lag_1_ratio_capacidad"]).any()


def test_generar_features_p3_clustering(
    monkeypatch, df_nomenclatura_sintetica, df_viajes_validos_sinteticos
):
    """
    Caso normal: Verifica que generar_features_p3_clustering agregue correctamente el
    perfil por estación y calcule proporciones entre 0 y 1 y ratio_salidas_llegadas > 0.
    """
    monkeypatch.setattr(features, "cargar_nomenclatura", lambda: df_nomenclatura_sintetica)
    df_p3 = features.generar_features_p3_clustering(df_viajes_validos_sinteticos)

    assert len(df_p3) == len(df_nomenclatura_sintetica)
    assert (df_p3["promedio_salidas_diarias"] > 0).all()
    assert df_p3["prop_fin_de_semana"].between(0.0, 1.0).all()
    assert df_p3["prop_pico_manana"].between(0.0, 1.0).all()
    assert (df_p3["ratio_salidas_llegadas"] > 0).all()


# ==============================================================================
# BLOQUE 3: PRUEBAS UNITARIAS DE VISUALIZACIÓN Y ARGUMENTOS (plots.py y train.py)
# ==============================================================================
def test_funciones_graficas_crean_archivos_png(tmp_path: Path):
    """Verifica que las funciones de plots.py generen archivos PNG válidos en disco."""
    y_real_reg = pd.Series([1.0, 3.0, 5.0, 2.0, 4.0, 6.0])
    y_pred_reg = np.array([1.2, 2.8, 4.9, 2.1, 3.9, 5.8])

    ruta_p1_a = plots.graficar_real_vs_predicho_p1(
        y_real_reg, y_pred_reg, tmp_path / "p1_real_pred.png"
    )
    ruta_p1_b = plots.graficar_residuales_p1(
        y_real_reg, y_pred_reg, tmp_path / "p1_res.png"
    )
    assert ruta_p1_a.exists() and ruta_p1_a.stat().st_size > 0
    assert ruta_p1_b.exists() and ruta_p1_b.stat().st_size > 0

    y_real_clf = pd.Series([0, 1, 0, 1, 1, 0])
    y_pred_clf = np.array([0, 1, 0, 0, 1, 0])
    y_prob_clf = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.15])

    ruta_p2_a = plots.graficar_matriz_confusion_p2(
        y_real_clf, y_pred_clf, tmp_path / "p2_cm.png"
    )
    ruta_p2_b = plots.graficar_curva_precision_recall_p2(
        y_real_clf, y_prob_clf, tmp_path / "p2_pr.png"
    )
    assert ruta_p2_a.exists() and ruta_p2_a.stat().st_size > 0
    assert ruta_p2_b.exists() and ruta_p2_b.stat().st_size > 0


def test_leer_argumentos_y_construir_estimador_casos_limite():
    """
    Caso normal y límite: Verifica que leer_argumentos reemplace 'from_config' por las
    constantes de config.py y que construir_estimador lance ValueError ante modelos inválidos.
    """
    args_defecto = train.leer_argumentos([])
    assert args_defecto.question == config.DEFAULT_QUESTION
    assert args_defecto.stage == config.DEFAULT_STAGE
    assert args_defecto.alpha == config.DEFAULT_ALPHA

    args_custom = train.leer_argumentos(
        ["--question", "p2", "--stage", "baseline", "--model", "from_config"]
    )
    assert args_custom.question == "p2"
    assert args_custom.model == "dummy"

    with pytest.raises(ValueError, match="Modelo no reconocido"):
        train.construir_estimador(question="p1", stage="candidate", model="modelo_inexistente")

    with pytest.raises(ValueError, match="Pregunta no reconocida"):
        train.construir_estimador(question="p99")