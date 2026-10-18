"""
Módulo de funciones de visualización para las figuras de P1, P2 y P3 del proyecto MiBici.
Separa la creación de gráficos para mantener el código ordenado y fácil de entender.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def graficar_real_vs_predicho_p1(
    y_real: pd.Series, y_pred: np.ndarray, ruta_salida: Path, semilla: int = 42
) -> Path:
    """Genera y guarda el gráfico de dispersión Real vs. Predicho para P1 (Regresión)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    muestra_idx = np.random.default_rng(semilla).choice(
        len(y_real), size=min(2000, len(y_real)), replace=False
    )
    plt.scatter(
        y_real.iloc[muestra_idx],
        y_pred[muestra_idx],
        alpha=0.3,
        color="#1f77b4",
        label="Predicciones (Muestra Test)",
    )
    valor_max = float(max(y_real.iloc[muestra_idx].max(), y_pred[muestra_idx].max()))
    plt.plot([0, valor_max], [0, valor_max], "r--", label="Identidad (Ideal)")
    plt.title("P1 Regresión (Ridge): Viajes Iniciados Reales vs. Predichos (Test)")
    plt.xlabel("Viajes Iniciados Reales por Hora")
    plt.ylabel("Viajes Iniciados Predichos por Hora")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


def graficar_matriz_confusion_p2(
    y_real: pd.Series, y_pred: np.ndarray, ruta_salida: Path
) -> Path:
    """Genera y guarda la Matriz de Confusión para P2 (Clasificación)."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    matriz = confusion_matrix(y_real, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz, display_labels=["Sin Riesgo (0)", "En Riesgo (1)"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    plt.title("P2 Clasificación (Logística): Matriz de Confusión (Test)", fontsize=11)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida


def graficar_mapa_clusters_p3(df_p3: pd.DataFrame, ruta_salida: Path) -> Path:
    """Genera y guarda el mapa geográfico de estaciones coloreado por Cluster para P3."""
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    for num_cluster in sorted(df_p3["cluster"].unique()):
        sub = df_p3.loc[df_p3["cluster"] == num_cluster]
        plt.scatter(
            sub["longitude"],
            sub["latitude"],
            s=35,
            alpha=0.8,
            label=f"Cluster {num_cluster} (n={len(sub)})",
        )
    plt.title("P3 Clustering (K-Means K=4): Distribución Geográfica de Estaciones")
    plt.xlabel("Longitud")
    plt.ylabel("Latitud")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(ruta_salida, dpi=150)
    plt.close()
    return ruta_salida