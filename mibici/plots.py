"""
Módulo de funciones de visualización para las figuras de P1, P2 y P3 del proyecto MiBici.
Genera dos gráficas de desempeño por pregunta para registrarlas como artefactos
en la subcarpeta 'plots/' dentro de MLflow Tracking.
"""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


# ==============================================================================
# GRÁFICAS PARA P1 (REGRESIÓN): REAL VS. PREDICHO Y ANÁLISIS DE RESIDUALES
# ==============================================================================
def graficar_real_vs_predicho_p1(
    y_real: pd.Series,
    y_pred: np.ndarray,
    ruta_salida: Path,
    semilla: int = 42,
    nombre_modelo: str = "Modelo",
) -> Path:
    """Genera y guarda el gráfico de dispersión Real vs. Predicho para P1 (Regresión)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))

    # Tomamos una muestra aleatoria de hasta 2,000 puntos para que el gráfico sea legible
    muestra_idx = np.random.default_rng(semilla).choice(
        len(y_real), size=min(2000, len(y_real)), replace=False
    )
    reales_muestra = y_real.iloc[muestra_idx]
    pred_muestra = y_pred[muestra_idx]

    plt.scatter(
        reales_muestra,
        pred_muestra,
        alpha=0.3,
        color="#1f77b4",
        label="Predicciones (Muestra Test)",
    )
    valor_max = float(max(reales_muestra.max(), pred_muestra.max()))
    plt.plot([0, valor_max], [0, valor_max], "r--", label="Identidad (Ideal)")
    plt.title(f"P1 Regresión ({nombre_modelo}): Reales vs. Predichos (Test)")
    plt.xlabel("Viajes Iniciados Reales por Hora")
    plt.ylabel("Viajes Iniciados Predichos por Hora")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


def graficar_residuales_p1(
    y_real: pd.Series,
    y_pred: np.ndarray,
    ruta_salida: Path,
    semilla: int = 42,
    nombre_modelo: str = "Modelo",
) -> Path:
    """Genera y guarda el gráfico de residuales (Real - Predicho) para P1 (Regresión)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    muestra_idx = np.random.default_rng(semilla).choice(
        len(y_real), size=min(2000, len(y_real)), replace=False
    )
    pred_muestra = y_pred[muestra_idx]
    residuales_muestra = y_real.iloc[muestra_idx] - pred_muestra

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Panel izquierdo: Residuales contra valores predichos
    axes[0].scatter(pred_muestra, residuales_muestra, alpha=0.3, color="#2ca02c")
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuales vs. Predichos ({nombre_modelo})")
    axes[0].set_xlabel("Viajes Predichos por Hora")
    axes[0].set_ylabel("Residual (Real - Predicho)")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    # Panel derecho: Histograma de los residuales
    axes[1].hist(residuales_muestra, bins=40, color="#1f77b4", edgecolor="black", alpha=0.75)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_title("Distribución del Error (Residuales en Test)")
    axes[1].set_xlabel("Residual (Real - Predicho)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


# ==============================================================================
# GRÁFICAS PARA P2 (CLASIFICACIÓN): MATRIZ DE CONFUSIÓN Y CURVA PRECISIÓN-RECALL
# ==============================================================================
def graficar_matriz_confusion_p2(
    y_real: pd.Series,
    y_pred: np.ndarray,
    ruta_salida: Path,
    nombre_modelo: str = "Modelo",
) -> Path:
    """Genera y guarda la Matriz de Confusión para P2 (Clasificación)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    matriz = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["Sin Riesgo (0)", "En Riesgo (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title(f"P2 Clasificación ({nombre_modelo}): Matriz de Confusión (Test)", fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


def graficar_curva_precision_recall_p2(
    y_real: pd.Series,
    y_prob: np.ndarray,
    ruta_salida: Path,
    nombre_modelo: str = "Modelo",
) -> Path:
    """Genera y guarda la Curva Precisión-Recall para P2 (Clasificación)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y_real, y_prob)
    ap_score = average_precision_score(y_real, y_prob)
    tasa_base = float(y_real.mean())

    plt.figure(figsize=(7.5, 5))
    plt.plot(
        recall,
        precision,
        color="#d62728",
        linewidth=2,
        label=f"{nombre_modelo} (AP = {ap_score:.4f})",
    )
    plt.axhline(
        tasa_base,
        color="gray",
        linestyle="--",
        label=f"Línea base azar (Prevalencia = {tasa_base:.4f})",
    )
    plt.title(f"P2 Clasificación ({nombre_modelo}): Curva Precisión-Recall (Test)")
    plt.xlabel("Recall (Sensibilidad sobre Estaciones en Riesgo)")
    plt.ylabel("Precision (Precisión de Alertas)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


# ==============================================================================
# GRÁFICAS PARA P3 (CLUSTERING): MAPA DE ESTACIONES Y PERFILES DE GRUPOS
# ==============================================================================
def graficar_mapa_clusters_p3(
    df_p3: pd.DataFrame,
    ruta_salida: Path,
    nombre_modelo: str = "KMeans",
) -> Path:
    """Genera y guarda el mapa geográfico de estaciones coloreado por Cluster para P3."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for num_cluster in sorted(df_p3["cluster"].unique()):
        sub = df_p3.loc[df_p3["cluster"] == num_cluster]
        plt.scatter(
            sub["longitude"],
            sub["latitude"],
            s=35,
            alpha=0.85,
            label=f"Cluster {num_cluster} (n={len(sub)})",
        )
    plt.title(f"P3 Clustering ({nombre_modelo}): Distribución Geográfica de Estaciones")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


def graficar_perfiles_clusters_p3(
    df_p3: pd.DataFrame,
    columnas_perfil: list[str],
    ruta_salida: Path,
    nombre_modelo: str = "KMeans",
) -> Path:
    """
    Genera y guarda una gráfica de barras comparando el perfil promedio (z-score)
    de las variables operativas en cada grupo de estaciones para P3.
    """
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    df_copia = df_p3.copy()

    # Estandarizamos cada variable para poder compararlas en la misma escala (media 0, std 1)
    for col in columnas_perfil:
        desv = df_copia[col].std()
        if desv > 0:
            df_copia[col] = (df_copia[col] - df_copia[col].mean()) / desv
        else:
            df_copia[col] = 0.0

    promedios_cluster = df_copia.groupby("cluster")[columnas_perfil].mean()

    ax = promedios_cluster.T.plot(kind="bar", figsize=(10, 5.5), width=0.8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title(f"P3 Clustering ({nombre_modelo}): Perfil Estandarizado por Grupo de Estaciones")
    plt.xlabel("Variables Operativas y Geográficas de la Estación")
    plt.ylabel("Promedio Estandarizado (Z-Score)")
    plt.xticks(rotation=30, ha="right")
    plt.legend(title="Cluster", loc="best")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida