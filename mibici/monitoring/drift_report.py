"""
Módulo de monitoreo de datos y generación de reportes de Data Drift y Data Quality
con Evidently para los modelos P1 (Regresión), P2 (Clasificación) y P3 (Clustering)
del proyecto MiBici (Equipo 35).

Cumple los Pasos 1, 2 y 3 de la actividad:
  1. Define el conjunto de referencia (Entrenamiento: Octubre 2022 - Diciembre 2025)
     y el conjunto actual de producción (Octubre 2026 con nomenclatura_2026_10.csv).
  2. Ejecuta Evidently (Report con DataDriftPreset y DataSummaryPreset) para evaluar
     el drift por variable, el drift en la variable objetivo/predicción y la calidad
     de los datos (conteos, valores nulos, duplicados y cambios de estaciones).
  3. Exporta los reportes interactivos en HTML, el resumen cuantitativo en JSON y
     las gráficas en PNG para el documento PDF y el README del repositorio.

Cómo ejecutarlo:
    python -m mibici.monitoring.drift_report
"""

import json
import warnings
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici import config, dataset, features
from mibici.api import model_loader

matplotlib.use("Agg")

# ==============================================================================
# 1. LISTAS DE VARIABLES DE CADA MODELO DEL PROYECTO MIBICI
# ==============================================================================
COLUMNAS_FEATURES_P1_P2 = [
    "hora",
    "dia_semana",
    "mes",
    "es_fin_de_semana",
    "es_hora_pico",
    "dpcapacity",
    "latitude",
    "longitude",
    "lag_1_iniciados",
    "lag_2_iniciados",
    "lag_24_iniciados",
    "media_movil_3_iniciados",
    "lag_1_terminados",
    "lag_1_flujo_neto",
    "lag_1_ratio_capacidad",
]

COLUMNAS_FEATURES_P3 = [
    "promedio_salidas_diarias",
    "duracion_mediana_min",
    "prop_fin_de_semana",
    "prop_pico_manana",
    "prop_pico_tarde",
    "ratio_salidas_llegadas",
    "dpcapacity",
    "latitude",
    "longitude",
]


# ==============================================================================
# 2. CONSTRUIR O CARGAR LAS TABLAS DE FEATURES DEL PERIODO ACTUAL (OCTUBRE 2026)
# ==============================================================================
def construir_features_octubre_2026() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Asegura que el lote de producción de Octubre 2026 esté limpio y versionado,
    y construye (con caché en data/interim/) las dos tablas del periodo actual:
      - features_p1_p2_2026_10.parquet (observaciones Estación x Hora de Octubre 2026)
      - features_p3_2026_10.parquet    (perfiles operativos por estación en Octubre 2026)
    """
    manifiesto_10 = dataset.asegurar_y_construir_lote_octubre_2026()
    config.INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    columnas_viajes = [
        "Viaje_Id",
        "Inicio_del_viaje",
        "Fin_del_viaje",
        "Origen_Id",
        "Destino_Id",
    ]

    # A) Tabla horaria de P1 y P2 para Octubre 2026
    if config.INTERIM_P1_P2_2026_10_FILE.exists():
        df_p1_p2_oct = pd.read_parquet(config.INTERIM_P1_P2_2026_10_FILE)
    else:
        print("Construyendo tabla horaria P1/P2 para el periodo actual (Octubre 2026)...")
        df_viajes_10 = pd.read_parquet(
            config.PARQUET_FILE_2026_10, columns=columnas_viajes
        )

        # Para que las primeras 24 horas del 1 de octubre de 2026 tengan sus rezagos
        # completos (lag_1, lag_2, lag_24), concatenamos los últimos 7 días de septiembre
        # de 2026 antes de calcular los .shift() y luego filtramos solo octubre de 2026.
        if config.PARQUET_FILE.exists():
            df_hist = pd.read_parquet(config.PARQUET_FILE, columns=columnas_viajes)
            corte_contexto = pd.Timestamp("2026-09-24 00:00:00")
            df_contexto_sep = df_hist.loc[df_hist["Inicio_del_viaje"] >= corte_contexto]
            df_unido = pd.concat([df_contexto_sep, df_viajes_10], ignore_index=True)
        else:
            df_unido = df_viajes_10

        tabla_horaria_10 = features.construir_tabla_horaria_estaciones(
            df_unido,
            ruta_nomenclatura=config.NOMENCLATURE_FILE_2026_10,
        )
        df_features_10 = features.generar_features_p1_y_p2(tabla_horaria_10)

        inicio_oct = pd.Timestamp(config.CURRENT_START_DATE)
        fin_oct = pd.Timestamp(config.CURRENT_END_DATE)
        df_p1_p2_oct = (
            df_features_10.loc[
                (df_features_10["Fecha_Hora"] >= inicio_oct)
                & (df_features_10["Fecha_Hora"] <= fin_oct)
            ]
            .reset_index(drop=True)
        )
        df_p1_p2_oct.to_parquet(config.INTERIM_P1_P2_2026_10_FILE, index=False)

    # B) Tabla de perfiles por estación de P3 para Octubre 2026
    if config.INTERIM_P3_2026_10_FILE.exists():
        df_p3_oct = pd.read_parquet(config.INTERIM_P3_2026_10_FILE)
    else:
        print("Construyendo perfiles de estaciones P3 para Octubre 2026...")
        df_viajes_10 = pd.read_parquet(
            config.PARQUET_FILE_2026_10, columns=columnas_viajes
        )
        df_p3_oct = features.generar_features_p3_clustering(
            df_viajes_10,
            ruta_nomenclatura=config.NOMENCLATURE_FILE_2026_10,
        )
        df_p3_oct.to_parquet(config.INTERIM_P3_2026_10_FILE, index=False)

    return df_p1_p2_oct, df_p3_oct, manifiesto_10


# ==============================================================================
# 3. PREPARAR CONJUNTOS DE REFERENCIA Y ACTUAL CON PREDICCIONES DEL MODELO v2
# ==============================================================================
def preparar_conjuntos_referencia_y_actual(
    tamano_muestra: int = config.EVIDENTLY_SAMPLE_SIZE,
) -> dict:
    """
    Carga el conjunto de referencia (Entrenamiento: Oct 2022 - Dic 2025) y el conjunto
    actual (Producción: Octubre 2026), agrega las predicciones de la versión vigente (v2)
    de los modelos P1, P2 y P3, y extrae una muestra determinista (semilla 42) para que
    los reportes de Evidently comparen muestras del mismo tamaño sin sesgo.
    """
    df_p1_p2_oct, df_p3_oct, manifiesto_10 = construir_features_octubre_2026()

    # 1. Cargamos la tabla histórica P1/P2 y filtramos el periodo de entrenamiento (Referencia)
    df_p1_p2_hist = pd.read_parquet(config.INTERIM_P1_P2_FILE)
    corte_train = pd.Timestamp(config.TRAIN_END_DATE)
    df_ref_completo = df_p1_p2_hist.loc[df_p1_p2_hist["Fecha_Hora"] <= corte_train].copy()

    # Tomamos una muestra aleatoria determinista (RANDOM_SEED = 42) para Evidently
    n_ref = min(tamano_muestra, len(df_ref_completo))
    n_cur = min(tamano_muestra, len(df_p1_p2_oct))
    df_ref_muestra = df_ref_completo.sample(
        n=n_ref, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)
    df_cur_muestra = df_p1_p2_oct.sample(
        n=n_cur, random_state=config.RANDOM_SEED
    ).reset_index(drop=True)

    # 2. Cargamos los modelos actuales en producción (v2 / @champion) para evaluar también
    #    el drift en las predicciones (Prediction Drift) y en las variables objetivo (Target Drift)
    modelo_p1, _ = model_loader.cargar_modelo_para_api("p1")
    modelo_p2, _ = model_loader.cargar_modelo_para_api("p2")
    modelo_p3, _ = model_loader.cargar_modelo_para_api("p3")

    columnas_analisis_p1_p2 = (
        COLUMNAS_FEATURES_P1_P2
        + ["viajes_iniciados", "en_riesgo", "pred_p1_viajes", "pred_p2_riesgo"]
    )

    X_ref_p1_p2 = df_ref_muestra[COLUMNAS_FEATURES_P1_P2].astype("float64")
    X_cur_p1_p2 = df_cur_muestra[COLUMNAS_FEATURES_P1_P2].astype("float64")

    df_ref_evidently_p1_p2 = X_ref_p1_p2.copy()
    df_ref_evidently_p1_p2["viajes_iniciados"] = df_ref_muestra["viajes_iniciados"].astype("float64")
    df_ref_evidently_p1_p2["en_riesgo"] = df_ref_muestra["en_riesgo"].astype("float64")
    df_ref_evidently_p1_p2["pred_p1_viajes"] = np.clip(
        modelo_p1.predict(X_ref_p1_p2), 0.0, None
    ).astype("float64")
    df_ref_evidently_p1_p2["pred_p2_riesgo"] = modelo_p2.predict(X_ref_p1_p2).astype("float64")

    df_cur_evidently_p1_p2 = X_cur_p1_p2.copy()
    df_cur_evidently_p1_p2["viajes_iniciados"] = df_cur_muestra["viajes_iniciados"].astype("float64")
    df_cur_evidently_p1_p2["en_riesgo"] = df_cur_muestra["en_riesgo"].astype("float64")
    df_cur_evidently_p1_p2["pred_p1_viajes"] = np.clip(
        modelo_p1.predict(X_cur_p1_p2), 0.0, None
    ).astype("float64")
    df_cur_evidently_p1_p2["pred_p2_riesgo"] = modelo_p2.predict(X_cur_p1_p2).astype("float64")

    # 3. Conjuntos de Referencia y Actual para P3 (Clustering de estaciones)
    df_p3_ref_raw = pd.read_parquet(config.INTERIM_P3_FILE)
    X_ref_p3 = df_p3_ref_raw[COLUMNAS_FEATURES_P3].astype("float64")
    X_cur_p3 = df_p3_oct[COLUMNAS_FEATURES_P3].astype("float64")

    df_ref_evidently_p3 = X_ref_p3.copy()
    df_ref_evidently_p3["cluster_asignado"] = modelo_p3.predict(X_ref_p3).astype("float64")

    df_cur_evidently_p3 = X_cur_p3.copy()
    df_cur_evidently_p3["cluster_asignado"] = modelo_p3.predict(X_cur_p3).astype("float64")

    return {
        "ref_p1_p2": df_ref_evidently_p1_p2[columnas_analisis_p1_p2],
        "cur_p1_p2": df_cur_evidently_p1_p2[columnas_analisis_p1_p2],
        "ref_p1_p2_total_rows": int(len(df_ref_completo)),
        "cur_p1_p2_total_rows": int(len(df_p1_p2_oct)),
        "ref_p3": df_ref_evidently_p3,
        "cur_p3": df_cur_evidently_p3,
        "manifiesto_2026_10": manifiesto_10,
    }


# ==============================================================================
# 4. EJECUTAR REPORTE DE EVIDENTLY Y EXTRAER MÉTRICAS DE DRIFT Y CALIDAD
# ==============================================================================
def ejecutar_reporte_evidently(
    df_referencia: pd.DataFrame,
    df_actual: pd.DataFrame,
    ruta_html: Path,
) -> dict:
    """
    Ejecuta Report([DataDriftPreset(), DataSummaryPreset()], include_tests=True)
    comparando df_actual contra df_referencia, guarda el reporte HTML interactivo
    y devuelve un diccionario limpio con las métricas de drift por variable y de
    calidad de datos (conteo de filas, nulos y duplicados).
    """
    ruta_html.parent.mkdir(parents=True, exist_ok=True)

    # En la versión actual de Evidently (0.7.x), report.run recibe (current_data, reference_data)
    reporte = Report(
        [DataDriftPreset(), DataSummaryPreset()],
        include_tests=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in divide",
            category=RuntimeWarning,
        )
        snapshot = reporte.run(df_actual, df_referencia)
    snapshot.save_html(str(ruta_html))

    diccionario_raw = snapshot.dict()
    metricas_raw = diccionario_raw.get("metrics", [])
    pruebas_raw = diccionario_raw.get("tests", [])

    # Mapa de estatus por columna a partir de las pruebas automáticas de Evidently
    estatus_test_columna = {}
    for prueba in pruebas_raw:
        params_m = prueba.get("metric_config", {}).get("params", {})
        if params_m.get("type") == "evidently:metric_v2:ValueDrift":
            col = params_m.get("column")
            estatus_test_columna[col] = prueba.get("status", "")

    resumen_global_drift = {"drifted_columns_count": 0, "drifted_columns_share": 0.0}
    calidad_datos = {
        "reference_rows_analyzed": int(len(df_referencia)),
        "current_rows_analyzed": int(len(df_actual)),
        "total_columns": int(df_actual.shape[1]),
        "reference_missing_cells": int(df_referencia.isna().sum().sum()),
        "current_missing_cells": int(df_actual.isna().sum().sum()),
        "current_duplicated_rows": 0,
    }
    drift_por_variable = {}

    for metrica in metricas_raw:
        cfg = metrica.get("config", {})
        tipo_metrica = cfg.get("type", "")
        valor = metrica.get("value")

        if tipo_metrica == "evidently:metric_v2:DriftedColumnsCount" and isinstance(valor, dict):
            resumen_global_drift["drifted_columns_count"] = int(valor.get("count", 0))
            resumen_global_drift["drifted_columns_share"] = round(float(valor.get("share", 0.0)), 4)

        elif tipo_metrica == "evidently:metric_v2:DuplicatedRowCount":
            calidad_datos["current_duplicated_rows"] = int(float(valor or 0))

        elif tipo_metrica == "evidently:metric_v2:ValueDrift":
            col = cfg.get("column")
            metodo = cfg.get("method", "unknown")
            umbral = float(cfg.get("threshold", config.DRIFT_STAT_THRESHOLD))
            puntaje = float(valor)

            # En métodos basados en p-value (K-S p_value, Z-test p_value, chi-square p_value),
            # existe drift estadísticamente significativo cuando p_value < threshold (0.05).
            # En métodos basados en distancia (Wasserstein, Jensen-Shannon, PSI),
            # existe drift cuando distance >= threshold.
            estatus_prueba = estatus_test_columna.get(col, "")
            if estatus_prueba in ("FAIL", "SUCCESS"):
                tiene_drift = estatus_prueba == "FAIL"
            elif "p_value" in metodo.lower():
                tiene_drift = puntaje < umbral
            else:
                tiene_drift = puntaje >= umbral

            serie_ref = df_referencia[col].dropna().astype("float64")
            serie_cur = df_actual[col].dropna().astype("float64")
            media_ref = float(serie_ref.mean())
            media_cur = float(serie_cur.mean())
            cambio_relativo_pct = (
                ((media_cur - media_ref) / abs(media_ref)) * 100.0
                if abs(media_ref) > 1e-9
                else 0.0
            )

            drift_por_variable[col] = {
                "column": col,
                "method": metodo,
                "threshold": umbral,
                "score": round(puntaje, 6),
                "drift_detected": bool(tiene_drift),
                "ref_mean": round(media_ref, 4),
                "cur_mean": round(media_cur, 4),
                "ref_std": round(float(serie_ref.std()), 4),
                "cur_std": round(float(serie_cur.std()), 4),
                "ref_median": round(float(serie_ref.median()), 4),
                "cur_median": round(float(serie_cur.median()), 4),
                "mean_shift_pct": round(float(cambio_relativo_pct), 2),
                "ref_missing": int(df_referencia[col].isna().sum()),
                "cur_missing": int(df_actual[col].isna().sum()),
            }

    return {
        "html_report_path": str(ruta_html.relative_to(config.PROJ_ROOT) if ruta_html.is_relative_to(config.PROJ_ROOT) else ruta_html),
        "drift_overview": resumen_global_drift,
        "data_quality": calidad_datos,
        "columns": drift_por_variable,
    }


# ==============================================================================
# 5. GENERAR GRÁFICAS EN ALTA RESOLUCIÓN PARA EL REPORTE PDF Y EL README
# ==============================================================================
def generar_graficas_drift_evidently(
    resultado_p1_p2: dict,
    resultado_p3: dict,
    df_ref_p1_p2: pd.DataFrame,
    df_cur_p1_p2: pd.DataFrame,
) -> list[Path]:
    """
    Crea 3 gráficas claras y profesionales con los resultados de Evidently:
      1. drift_evidently_p1_p2_variables.png: Cambio relativo de medias (%) y detección
         de drift por variable en P1 y P2 (incluyendo targets y predicciones).
      2. drift_evidently_distribuciones_clave.png: Comparación de distribuciones entre
         Referencia y Actual (Octubre 2026) en las 4 variables operativas clave.
      3. drift_evidently_p3_perfiles.png: Cambio relativo de medias (%) y detección de
         drift en las variables del perfil de estaciones P3.
    """
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    rutas_generadas = []

    # --------------------------------------------------------------------------
    # GRÁFICA 1: Resumen de Drift por Variable en P1 y P2 (Evidently)
    # --------------------------------------------------------------------------
    ruta_g1 = config.FIGURES_DIR / "drift_evidently_p1_p2_variables.png"
    cols_p1_p2 = list(resultado_p1_p2["columns"].values())
    # Excluimos 'mes' de la barra de porcentaje para no distorsionar la escala visual
    # y lo anotamos explícitamente en el subtítulo
    cols_plot = [c for c in cols_p1_p2 if c["column"] != "mes"]
    cols_plot = sorted(cols_plot, key=lambda x: abs(x["mean_shift_pct"]))

    nombres = [c["column"] for c in cols_plot]
    cambios = [c["mean_shift_pct"] for c in cols_plot]
    colores = ["#d9534f" if c["drift_detected"] else "#2b8cbe" for c in cols_plot]

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    barras = ax.barh(nombres, cambios, color=colores, edgecolor="black", alpha=0.88)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")

    for barra, info_col in zip(barras, cols_plot):
        ancho = barra.get_width()
        desplazamiento = 1.2 if ancho >= 0 else -1.2
        alineacion = "left" if ancho >= 0 else "right"
        etiqueta = f"{ancho:+.1f}% (dist={info_col['score']:.3f})"
        ax.text(
            ancho + desplazamiento,
            barra.get_y() + barra.get_height() / 2.0,
            etiqueta,
            va="center",
            ha=alineacion,
            fontsize=8.5,
            fontweight="bold" if info_col["drift_detected"] else "normal",
        )

    conteo_drift = resultado_p1_p2["drift_overview"]["drifted_columns_count"]
    total_cols = resultado_p1_p2["data_quality"]["total_columns"]
    ax.set_title(
        f"Evidently Data & Prediction Drift — Modelos P1 y P2 "
        f"(Referencia Oct 2022–Dic 2025 vs. Actual Oct 2026)\n"
        f"Columnas con drift significativo: {conteo_drift}/{total_cols} "
        f"(Rojo = Drift detectado dist >= 0.10 | Azul = Estable)",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Cambio porcentual en la media del periodo Actual vs. Referencia (%)")
    ax.set_xlim(min(cambios) - 18, max(cambios) + 22)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(ruta_g1)
    plt.close(fig)
    rutas_generadas.append(ruta_g1)

    # --------------------------------------------------------------------------
    # GRÁFICA 2: Distribuciones de las 4 Variables Clave con Drift (P1 y P2)
    # --------------------------------------------------------------------------
    ruta_g2 = config.FIGURES_DIR / "drift_evidently_distribuciones_clave.png"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    # Panel 1: Demanda horaria (viajes_iniciados, truncado al percentil 99 para visualización)
    ax1 = axes[0, 0]
    bins_viajes = np.arange(0, 16, 1)
    ax1.hist(
        np.clip(df_ref_p1_p2["viajes_iniciados"], 0, 15),
        bins=bins_viajes,
        density=True,
        alpha=0.55,
        label=f"Referencia (media={df_ref_p1_p2['viajes_iniciados'].mean():.2f})",
        color="#2b8cbe",
    )
    ax1.hist(
        np.clip(df_cur_p1_p2["viajes_iniciados"], 0, 15),
        bins=bins_viajes,
        density=True,
        alpha=0.55,
        label=f"Actual Oct 2026 (media={df_cur_p1_p2['viajes_iniciados'].mean():.2f})",
        color="#d9534f",
    )
    ax1.set_title("Distribución de Demanda Horaria (viajes_iniciados - P1)", fontweight="bold")
    ax1.set_xlabel("Viajes iniciados por estación-hora")
    ax1.set_ylabel("Densidad")
    ax1.legend(fontsize=8.5)
    ax1.grid(alpha=0.3)

    # Panel 2: Capacidad de estaciones (dpcapacity tras actualización nomenclatura_2026_10)
    ax2 = axes[0, 1]
    bins_cap = np.arange(10, 42, 2)
    ax2.hist(
        df_ref_p1_p2["dpcapacity"],
        bins=bins_cap,
        density=True,
        alpha=0.55,
        label=f"Referencia 2026_09 (media={df_ref_p1_p2['dpcapacity'].mean():.2f})",
        color="#2b8cbe",
    )
    ax2.hist(
        df_cur_p1_p2["dpcapacity"],
        bins=bins_cap,
        density=True,
        alpha=0.55,
        label=f"Actual 2026_10 (media={df_cur_p1_p2['dpcapacity'].mean():.2f})",
        color="#d9534f",
    )
    ax2.set_title("Capacidad de Estación (dpcapacity - Oct 2026)", fontweight="bold")
    ax2.set_xlabel("Capacidad de anclajes (dpcapacity)")
    ax2.set_ylabel("Densidad")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    # Panel 3: Promedio móvil de 3 horas de viajes iniciados
    ax3 = axes[1, 0]
    bins_mm = np.linspace(0, 12, 25)
    ax3.hist(
        np.clip(df_ref_p1_p2["media_movil_3_iniciados"], 0, 12),
        bins=bins_mm,
        density=True,
        alpha=0.55,
        label=f"Referencia (media={df_ref_p1_p2['media_movil_3_iniciados'].mean():.2f})",
        color="#2b8cbe",
    )
    ax3.hist(
        np.clip(df_cur_p1_p2["media_movil_3_iniciados"], 0, 12),
        bins=bins_mm,
        density=True,
        alpha=0.55,
        label=f"Actual Oct 2026 (media={df_cur_p1_p2['media_movil_3_iniciados'].mean():.2f})",
        color="#d9534f",
    )
    ax3.set_title("Tendencia Reciente (media_movil_3_iniciados)", fontweight="bold")
    ax3.set_xlabel("Promedio móvil 3h de viajes iniciados")
    ax3.set_ylabel("Densidad")
    ax3.legend(fontsize=8.5)
    ax3.grid(alpha=0.3)

    # Panel 4: Tasa de riesgo real vs predicha en P2 (en_riesgo y pred_p2_riesgo)
    ax4 = axes[1, 1]
    categorias = ["Tasa Real\nen_riesgo", "Tasa Predicha\nModelo v2 (P2)"]
    valores_ref = [
        df_ref_p1_p2["en_riesgo"].mean() * 100.0,
        df_ref_p1_p2["pred_p2_riesgo"].mean() * 100.0,
    ]
    valores_cur = [
        df_cur_p1_p2["en_riesgo"].mean() * 100.0,
        df_cur_p1_p2["pred_p2_riesgo"].mean() * 100.0,
    ]
    x_pos = np.arange(len(categorias))
    ancho_b = 0.34
    ax4.bar(
        x_pos - ancho_b / 2,
        valores_ref,
        width=ancho_b,
        label="Referencia (Train)",
        color="#2b8cbe",
        edgecolor="black",
    )
    ax4.bar(
        x_pos + ancho_b / 2,
        valores_cur,
        width=ancho_b,
        label="Actual (Octubre 2026)",
        color="#d9534f",
        edgecolor="black",
    )
    for idx_b, (vr, vc) in enumerate(zip(valores_ref, valores_cur)):
        ax4.text(idx_b - ancho_b / 2, vr + 0.4, f"{vr:.2f}%", ha="center", fontsize=8.5)
        ax4.text(idx_b + ancho_b / 2, vc + 0.4, f"{vc:.2f}%", ha="center", fontsize=8.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(categorias)
    ax4.set_ylabel("Porcentaje de horas en riesgo (%)")
    ax4.set_title("Prevalencia de Riesgo de Desbalance >=30% (P2)", fontweight="bold")
    ax4.set_ylim(0, max(valores_ref + valores_cur) + 5.0)
    ax4.legend(fontsize=8.5)
    ax4.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Comparación de Distribuciones en Variables Afectadas por Drift (Evidently)",
        fontsize=12.5,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(ruta_g2)
    plt.close(fig)
    rutas_generadas.append(ruta_g2)

    # --------------------------------------------------------------------------
    # GRÁFICA 3: Resumen de Drift en Perfiles de Estaciones P3 (Clustering)
    # --------------------------------------------------------------------------
    ruta_g3 = config.FIGURES_DIR / "drift_evidently_p3_perfiles.png"
    cols_p3 = list(resultado_p3["columns"].values())
    cols_p3 = sorted(cols_p3, key=lambda x: abs(x["mean_shift_pct"]))

    nombres_p3 = [c["column"] for c in cols_p3]
    cambios_p3 = [c["mean_shift_pct"] for c in cols_p3]
    colores_p3 = ["#d9534f" if c["drift_detected"] else "#2b8cbe" for c in cols_p3]

    fig, ax = plt.subplots(figsize=(10.5, 5.5), dpi=150)
    barras_p3 = ax.barh(nombres_p3, cambios_p3, color=colores_p3, edgecolor="black", alpha=0.88)
    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="--")

    for barra, info_col in zip(barras_p3, cols_p3):
        ancho = barra.get_width()
        desp = 0.8 if ancho >= 0 else -0.8
        alin = "left" if ancho >= 0 else "right"
        ax.text(
            ancho + desp,
            barra.get_y() + barra.get_height() / 2.0,
            f"{ancho:+.1f}% (p={info_col['score']:.3f})",
            va="center",
            ha=alin,
            fontsize=8.5,
            fontweight="bold" if info_col["drift_detected"] else "normal",
        )

    c_drift_p3 = resultado_p3["drift_overview"]["drifted_columns_count"]
    t_cols_p3 = resultado_p3["data_quality"]["total_columns"]
    ax.set_title(
        f"Evidently Data Drift — Perfiles de Estaciones P3 (Clustering)\n"
        f"Referencia (373 estaciones Train) vs. Actual Octubre 2026 (469 estaciones activas) | "
        f"Drift: {c_drift_p3}/{t_cols_p3} columnas",
        fontsize=9.5,
        fontweight="bold",
    )
    ax.set_xlabel("Cambio porcentual en la media del perfil por estación (%)")
    ax.set_xlim(min(cambios_p3) - 12, max(cambios_p3) + 16)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(ruta_g3)
    plt.close(fig)
    rutas_generadas.append(ruta_g3)

    return rutas_generadas




def registrar_monitoreo_en_mlflow(resumen_completo: dict) -> None:
    """
    Registra en el servidor de MLflow las métricas de Data Drift y sube como artefactos
    los reportes HTML interactivos de Evidently y las gráficas PNG (Módulo 3).
    """
    import mlflow
    try:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        with mlflow.start_run(run_name="Evidently_DataDrift_Monitoring_Oct2026"):
            mlflow.set_tags(
                {
                    "team": config.TEAM_TAG,
                    "stage_of_work": "monitoring_drift",
                    "reference_period": "2022-10_a_2025-12",
                    "current_period": "2026-10",
                    "data_version_current": resumen_completo["current_period"]["data_version"],
                }
            )
            p1_ov = resumen_completo["p1_p2_hourly_monitoring"]["drift_overview"]
            p3_ov = resumen_completo["p3_station_profile_monitoring"]["drift_overview"]
            mlflow.log_metrics(
                {
                    "p1_p2_drifted_columns_count": float(p1_ov["drifted_columns_count"]),
                    "p1_p2_drifted_columns_share": float(p1_ov["drifted_columns_share"]),
                    "p3_drifted_columns_count": float(p3_ov["drifted_columns_count"]),
                    "p3_drifted_columns_share": float(p3_ov["drifted_columns_share"]),
                    "oct2026_clean_trips": float(resumen_completo["current_period"]["clean_trips_count"]),
                }
            )
            mlflow.log_artifact(str(config.EVIDENTLY_HTML_P1_P2), artifact_path="monitoring")
            mlflow.log_artifact(str(config.EVIDENTLY_HTML_P3), artifact_path="monitoring")
            mlflow.log_artifact(str(config.DRIFT_SUMMARY_JSON), artifact_path="monitoring")
    except Exception as exc:
        print(f"Aviso: No se pudo registrar corrida de monitoreo en MLflow ({exc}).")


# ==============================================================================
# 6. FLUJO PRINCIPAL DE MONITOREO DE DRIFT CON EVIDENTLY
# ==============================================================================
def ejecutar_monitoreo_completo() -> dict:
    """
    Orquesta de principio a fin la preparación de los conjuntos de referencia y actual,
    la ejecución de Evidently para P1/P2 y P3, la generación de las gráficas PNG y el
    guardado del resumen JSON para el reporte técnico.
    """
    print("=" * 80)
    print("INICIANDO MONITOREO DE DATA DRIFT Y CALIDAD DE DATOS CON EVIDENTLY (EQUIPO 35)")
    print("=" * 80)

    datos = preparar_conjuntos_referencia_y_actual()

    print("\n1. Ejecutando Evidently para P1 (Regresión) y P2 (Clasificación)...")
    res_p1_p2 = ejecutar_reporte_evidently(
        df_referencia=datos["ref_p1_p2"],
        df_actual=datos["cur_p1_p2"],
        ruta_html=config.EVIDENTLY_HTML_P1_P2,
    )
    res_p1_p2["data_quality"]["reference_total_rows_population"] = datos["ref_p1_p2_total_rows"]
    res_p1_p2["data_quality"]["current_total_rows_population"] = datos["cur_p1_p2_total_rows"]

    print("2. Ejecutando Evidently para P3 (Clustering de perfiles de estaciones)...")
    res_p3 = ejecutar_reporte_evidently(
        df_referencia=datos["ref_p3"],
        df_actual=datos["cur_p3"],
        ruta_html=config.EVIDENTLY_HTML_P3,
    )

    print("3. Generando gráficas PNG de distribuciones y métricas de Evidently...")
    rutas_png = generar_graficas_drift_evidently(
        resultado_p1_p2=res_p1_p2,
        resultado_p3=res_p3,
        df_ref_p1_p2=datos["ref_p1_p2"],
        df_cur_p1_p2=datos["cur_p1_p2"],
    )

    resumen_completo = {
        "team": config.TEAM_TAG,
        "evidently_version": "0.7.23",
        "reference_period": {
            "label": "Entrenamiento (Octubre 2022 - Diciembre 2025)",
            "months": 39,
            "data_version": "128979137b46",
            "nomenclature_file": "nomenclatura_2026_09.csv",
        },
        "current_period": {
            "label": "Producción (Octubre 2026)",
            "months": 1,
            "data_version": datos["manifiesto_2026_10"]["data_version_short"],
            "nomenclature_file": "nomenclatura_2026_10.csv",
            "clean_trips_count": datos["manifiesto_2026_10"]["rows_written_clean"],
            "nomenclature_changes": datos["manifiesto_2026_10"]["nomenclature_file"],
        },
        "p1_p2_hourly_monitoring": res_p1_p2,
        "p3_station_profile_monitoring": res_p3,
        "generated_figures": [str(p.relative_to(config.PROJ_ROOT)) for p in rutas_png],
    }

    config.MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DRIFT_SUMMARY_JSON, "w", encoding="utf-8") as f_json:
        json.dump(resumen_completo, f_json, indent=4, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("RESUMEN DE RESULTADOS DE EVIDENTLY")
    print("=" * 80)
    print(
        f"  * Reporte HTML P1/P2: {res_p1_p2['html_report_path']} "
        f"(Drift en {res_p1_p2['drift_overview']['drifted_columns_count']}/"
        f"{res_p1_p2['data_quality']['total_columns']} columnas, "
        f"share={res_p1_p2['drift_overview']['drifted_columns_share']:.2%})"
    )
    for col_name, info_c in res_p1_p2["columns"].items():
        estado = "DRIFT" if info_c["drift_detected"] else "ESTABLE"
        print(
            f"      - {col_name:24s}: {estado:7s} | método={info_c['method']:14s} | "
            f"score={info_c['score']:.4f} | media_ref={info_c['ref_mean']:.4f} -> "
            f"media_act={info_c['cur_mean']:.4f} ({info_c['mean_shift_pct']:+.2f}%)"
        )

    print(
        f"\n  * Reporte HTML P3:    {res_p3['html_report_path']} "
        f"(Drift en {res_p3['drift_overview']['drifted_columns_count']}/"
        f"{res_p3['data_quality']['total_columns']} columnas, "
        f"share={res_p3['drift_overview']['drifted_columns_share']:.2%})"
    )
    for col_name, info_c in res_p3["columns"].items():
        estado = "DRIFT" if info_c["drift_detected"] else "ESTABLE"
        print(
            f"      - {col_name:24s}: {estado:7s} | método={info_c['method']:14s} | "
            f"score={info_c['score']:.4f} | media_ref={info_c['ref_mean']:.4f} -> "
            f"media_act={info_c['cur_mean']:.4f} ({info_c['mean_shift_pct']:+.2f}%)"
        )
    print("=" * 80)
    registrar_monitoreo_en_mlflow(resumen_completo)
    return resumen_completo


if __name__ == "__main__":
    ejecutar_monitoreo_completo()