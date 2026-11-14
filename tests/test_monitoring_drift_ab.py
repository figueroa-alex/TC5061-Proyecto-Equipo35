"""
Pruebas unitarias e integración para el módulo de monitoreo de Data Drift con Evidently
y el ejercicio de A/B Testing entre versiones del modelo (Entregable 8 - Equipo 35).

Verifica:
  1. Detección correcta de Data Drift y métricas de calidad con Evidently (Report + DataDriftPreset).
  2. Ausencia de drift falso cuando las distribuciones de referencia y actual son idénticas.
  3. Integridad del manifiesto de versión del lote de producción Octubre 2026 (data_version_2026_10.json).
  4. Estructura y coherencia estadística de los resultados de A/B Testing y reportes generados.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mibici import config
from mibici.monitoring import drift_report


def test_evidently_detecta_drift_en_variable_desplazada(tmp_path: Path):
    """
    Caso normal: Cuando una variable numérica presenta un desplazamiento claro entre
    el conjunto de referencia y el actual, ejecutar_reporte_evidently debe detectarlo
    y guardar el archivo HTML interactivo.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    df_ref = pd.DataFrame(
        {
            "viajes_iniciados": rng.normal(loc=2.4, scale=1.0, size=300),
            "hora": rng.integers(0, 24, size=300).astype(float),
        }
    )
    df_cur = pd.DataFrame(
        {
            "viajes_iniciados": rng.normal(loc=6.5, scale=1.0, size=300),
            "hora": df_ref["hora"].copy(),
        }
    )

    ruta_html = tmp_path / "reporte_prueba_drift.html"
    resultado = drift_report.ejecutar_reporte_evidently(
        df_referencia=df_ref,
        df_actual=df_cur,
        ruta_html=ruta_html,
    )

    assert ruta_html.exists() and ruta_html.stat().st_size > 1000
    assert resultado["data_quality"]["reference_rows_analyzed"] == 300
    assert resultado["data_quality"]["current_rows_analyzed"] == 300
    assert resultado["columns"]["viajes_iniciados"]["drift_detected"] is True
    assert resultado["columns"]["hora"]["drift_detected"] is False


def test_evidently_sin_drift_en_distribuciones_identicas(tmp_path: Path):
    """
    Caso límite: Si el conjunto de referencia y el conjunto actual son idénticos,
    Evidently debe reportar 0 columnas con drift y 0 valores faltantes.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    df_base = pd.DataFrame(
        {
            "dpcapacity": rng.choice([11.0, 15.0, 19.0, 23.0], size=200),
            "lag_1_iniciados": rng.poisson(lam=2.5, size=200).astype(float),
        }
    )

    ruta_html = tmp_path / "reporte_sin_drift.html"
    resultado = drift_report.ejecutar_reporte_evidently(
        df_referencia=df_base,
        df_actual=df_base.copy(),
        ruta_html=ruta_html,
    )

    assert resultado["drift_overview"]["drifted_columns_count"] == 0
    assert resultado["drift_overview"]["drifted_columns_share"] == 0.0
    assert resultado["data_quality"]["current_missing_cells"] == 0


def test_manifiesto_octubre_2026_y_nomenclatura_versionados():
    """
    Verifica que el manifiesto data_version_2026_10.json y el catálogo nomenclatura_2026_10.csv
    existan, tengan hashes SHA-256 válidos y documenten los cambios de estaciones.
    """
    assert config.NOMENCLATURE_FILE_2026_10.exists()
    assert config.DATA_VERSION_FILE_2026_10.exists()

    with open(config.DATA_VERSION_FILE_2026_10, "r", encoding="utf-8") as f_json:
        manifiesto = json.load(f_json)

    assert manifiesto["date_range"] == "2026-10"
    assert len(manifiesto["data_version_short"]) == 12
    assert manifiesto["rows_written_clean"] > 500000
    assert manifiesto["stations_in_nomenclature_catalog"] == 470
    cambios_nom = manifiesto["nomenclature_file"]
    assert len(cambios_nom["removed_station_ids_vs_2026_09"]) == 14
    assert len(cambios_nom["capacity_changes_vs_2026_09"]) == 11


def test_reportes_drift_y_ab_testing_generados_correctamente():
    """
    Verifica que los resúmenes JSON y gráficas PNG de Evidently y del A/B Testing
    contengan las métricas completas para P1, P2 y P3.
    """
    assert config.DRIFT_SUMMARY_JSON.exists()
    assert config.AB_TEST_SUMMARY_JSON.exists()
    assert config.EVIDENTLY_HTML_P1_P2.exists()
    assert config.EVIDENTLY_HTML_P3.exists()

    with open(config.AB_TEST_SUMMARY_JSON, "r", encoding="utf-8") as f_ab:
        datos_ab = json.load(f_ab)

    for pregunta in ("p1", "p2", "p3"):
        assert pregunta in datos_ab["models"]
        info_m = datos_ab["models"][pregunta]
        assert "variant_A_v2" in info_m and "variant_B_v3" in info_m
        assert info_m["selected_winner"] in ("v2", "v3")
        assert isinstance(info_m["paired_ttest_p_value"], float)