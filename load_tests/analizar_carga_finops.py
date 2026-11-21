"""
Análisis de pruebas de carga (Locust), monitoreo de recursos (podman/docker stats)
y estimación FinOps del costo por inferencia para el proyecto MiBici (Equipo 35).

Entradas (generadas en Paso 6 dentro de reports/operations/):
  - locust_normal_stats.csv y locust_normal_stats_history.csv
  - locust_pico_stats.csv   y locust_pico_stats_history.csv
  - podman_stats_normal.csv y podman_stats_pico.csv

Salidas:
  - reports/figures/locust_desempeno_carga.png
  - reports/figures/recursos_contenedores_stats.png
  - reports/figures/finops_costo_inferencia.png
  - reports/operations/finops_summary_equipo35.json

Ejecución:
    python load_tests/analizar_carga_finops.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==============================================================================
# 1. RUTAS DE ENTRADA Y SALIDA
# ==============================================================================
PROJ_ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = PROJ_ROOT / "reports" / "operations"
FIG_DIR = PROJ_ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_LOCUST_NORMAL = OPS_DIR / "locust_normal_stats.csv"
CSV_LOCUST_PICO = OPS_DIR / "locust_pico_stats.csv"
CSV_HIST_PICO = OPS_DIR / "locust_pico_stats_history.csv"
CSV_STATS_NORMAL = OPS_DIR / "podman_stats_normal.csv"
CSV_STATS_PICO = OPS_DIR / "podman_stats_pico.csv"

FIG_LOCUST = FIG_DIR / "locust_desempeno_carga.png"
FIG_RECURSOS = FIG_DIR / "recursos_contenedores_stats.png"
FIG_FINOPS = FIG_DIR / "finops_costo_inferencia.png"
JSON_FINOPS = OPS_DIR / "finops_summary_equipo35.json"

# ==============================================================================
# 2. TARIFAS OFICIALES DE REFERENCIA EN LA NUBE (AWS EC2 LINUX - US-EAST-1)
# ==============================================================================
# Fuente: AWS EC2 Pricing (región us-east-1, mes estándar de 730 horas)
# - EBS gp3: $0.08 / GB-mes -> 20 GB para imágenes, Redis AOF y MLflow SQLite = $1.60 / mes
HORAS_MES = 730.0
SEGUNDOS_MES = HORAS_MES * 3600.0
EBS_GB = 20.0
EBS_COSTO_GB_MES = 0.08
EBS_COSTO_MES = EBS_GB * EBS_COSTO_GB_MES  # $1.60 USD/mes

ESCENARIOS_NUBE = [
    {
        "id": "baseline_t3_medium_od",
        "etiqueta": "Línea base\nt3.medium (OD)",
        "instancia": "t3.medium (2 vCPU, 4 GiB, x86)",
        "modalidad": "On-Demand 24/7",
        "tarifa_hora_usd": 0.0416,
        "vcpu": 2,
        "ram_gib": 4.0,
        "horas_mes": HORAS_MES,
    },
    {
        "id": "opt1_t3_small_od",
        "etiqueta": "Opt. 1: Rightsizing\nt3.small (OD)",
        "instancia": "t3.small (2 vCPU, 2 GiB, x86)",
        "modalidad": "On-Demand 24/7",
        "tarifa_hora_usd": 0.0208,
        "vcpu": 2,
        "ram_gib": 2.0,
        "horas_mes": HORAS_MES,
    },
    {
        "id": "opt2_t4g_small_od",
        "etiqueta": "Opt. 2: Graviton ARM\nt4g.small (OD)",
        "instancia": "t4g.small (2 vCPU, 2 GiB, ARM64)",
        "modalidad": "On-Demand 24/7",
        "tarifa_hora_usd": 0.0168,
        "vcpu": 2,
        "ram_gib": 2.0,
        "horas_mes": HORAS_MES,
    },
    {
        "id": "opt3_t4g_small_ri1y",
        "etiqueta": "Opt. 3: ARM + RI 1a\nt4g.small (RI 1 año)",
        "instancia": "t4g.small (2 vCPU, 2 GiB, ARM64)",
        "modalidad": "Reserved 1 año (No Upfront)",
        "tarifa_hora_usd": 0.0110,
        "vcpu": 2,
        "ram_gib": 2.0,
        "horas_mes": HORAS_MES,
    },
    {
        "id": "opt4_t4g_small_horario",
        "etiqueta": "Opt. 4: ARM + Horario\nt4g.small (18h/día)",
        "instancia": "t4g.small (2 vCPU, 2 GiB, ARM64)",
        "modalidad": "On-Demand programado (18 h/día)",
        "tarifa_hora_usd": 0.0168,
        "vcpu": 2,
        "ram_gib": 2.0,
        "horas_mes": 18.0 * 30.4167,  # 547.5 h/mes (apagado 6h en madrugada)
    },
]

# Endpoints que corresponden estrictamente a inferencia de modelos ML
ENDPOINTS_INFERENCIA = {
    "POST /predict/p1-demanda",
    "POST /predict/p2-riesgo",
    "POST /predict/p3-grupos",
    "POST /predict/async/p1-demanda",
}


# ==============================================================================
# 3. CARGA DE DATOS REALES DE LOCUST Y PODMAN STATS
# ==============================================================================
print("=" * 78)
print("ANALIZANDO RESULTADOS DE CARGA (LOCUST), RECURSOS (STATS) Y FINOPS - EQUIPO 35")
print("=" * 78)

df_locust_normal = pd.read_csv(CSV_LOCUST_NORMAL)
df_locust_pico = pd.read_csv(CSV_LOCUST_PICO)
df_hist_pico = pd.read_csv(CSV_HIST_PICO)
df_stats_normal = pd.read_csv(CSV_STATS_NORMAL)
df_stats_pico = pd.read_csv(CSV_STATS_PICO)

# Extraer filas agregadas ("Aggregated") de ambos escenarios
agg_normal = df_locust_normal[df_locust_normal["Name"] == "Aggregated"].iloc[0]
agg_pico = df_locust_pico[df_locust_pico["Name"] == "Aggregated"].iloc[0]

# Extraer solo los endpoints individuales (sin la fila Aggregated)
ep_pico = df_locust_pico[df_locust_pico["Name"] != "Aggregated"].copy()
ep_normal = df_locust_normal[df_locust_normal["Name"] != "Aggregated"].copy()

# Calcular RPS exclusivamente de endpoints de inferencia ML (excluyendo /health y /jobs)
rps_inferencia_normal = float(
    ep_normal[ep_normal["Name"].isin(ENDPOINTS_INFERENCIA)]["Requests/s"].sum()
)
rps_inferencia_pico = float(
    ep_pico[ep_pico["Name"].isin(ENDPOINTS_INFERENCIA)]["Requests/s"].sum()
)

# Resumen de consumo de recursos por contenedor (Normal y Pico)
def resumir_stats(df_stats: pd.DataFrame) -> dict:
    resumen = {}
    for nombre, grupo in df_stats.groupby("container_name"):
        resumen[nombre] = {
            "cpu_promedio_pct": round(float(grupo["cpu_percent"].mean()), 2),
            "cpu_pico_pct": round(float(grupo["cpu_percent"].max()), 2),
            "ram_promedio_mb": round(float(grupo["mem_usage_mb"].mean()), 2),
            "ram_pico_mb": round(float(grupo["mem_usage_mb"].max()), 2),
            "ram_limite_mb": round(float(grupo["mem_limit_mb"].max()), 2),
        }
    return resumen

recursos_normal = resumir_stats(df_stats_normal)
recursos_pico = resumir_stats(df_stats_pico)

# Totales combinados de los 4 microservicios en carga pico
ram_pico_total_mb = round(sum(v["ram_pico_mb"] for v in recursos_pico.values()), 2)
ram_prom_total_mb = round(sum(v["ram_promedio_mb"] for v in recursos_pico.values()), 2)
cpu_pico_total_pct = round(sum(v["cpu_pico_pct"] for v in recursos_pico.values()), 2)
cpu_prom_total_pct = round(sum(v["cpu_promedio_pct"] for v in recursos_pico.values()), 2)

# Consumo solo de la capa de inferencia (sin mlflow-server)
ram_inferencia_pico_mb = round(
    sum(v["ram_pico_mb"] for k, v in recursos_pico.items() if k != "mibici-mlflow"), 2
)


# ==============================================================================
# 4. GRÁFICA 1: DESEMPEÑO DE PRUEBAS DE CARGA CON LOCUST (2x2)
# ==============================================================================
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
fig.suptitle(
    "Pruebas de Carga con Locust — Arquitectura de Microservicios MiBici (Equipo 35)",
    fontsize=14,
    fontweight="bold",
    y=0.98,
)

# Preparar serie temporal del escenario pico
hist_agg = df_hist_pico[df_hist_pico["Name"] == "Aggregated"].copy()
hist_agg["segundos"] = hist_agg["Timestamp"] - hist_agg["Timestamp"].iloc[0]

# (a) RPS y usuarios concurrentes en el tiempo (Escenario Pico)
ax1 = axes[0, 0]
l1 = ax1.plot(
    hist_agg["segundos"],
    hist_agg["Requests/s"],
    color="#1f77b4",
    linewidth=2.2,
    label="Peticiones/s (RPS)",
)
ax1.set_title("A) Evolución de RPS y Usuarios (Escenario Pico: 50 usuarios)", fontweight="bold")
ax1.set_xlabel("Tiempo transcurrido (segundos)")
ax1.set_ylabel("Peticiones por segundo (RPS)", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")

ax1_twin = ax1.twinx()
l2 = ax1_twin.plot(
    hist_agg["segundos"],
    hist_agg["User Count"],
    color="#ff7f0e",
    linestyle="--",
    linewidth=2.0,
    label="Usuarios concurrentes",
)
ax1_twin.set_ylabel("Usuarios activos", color="#ff7f0e")
ax1_twin.tick_params(axis="y", labelcolor="#ff7f0e")
ax1_twin.grid(False)
lineas = l1 + l2
etiquetas = [ln.get_label() for ln in lineas]
ax1.legend(lineas, etiquetas, loc="lower right", frameon=True)

# (b) Latencias P50 y P95 en el tiempo (Escenario Pico)
ax2 = axes[0, 1]
p50_col = pd.to_numeric(hist_agg["50%"], errors="coerce")
p95_col = pd.to_numeric(hist_agg["95%"], errors="coerce")
ax2.plot(hist_agg["segundos"], p50_col, color="#2ca02c", linewidth=2.0, label="Mediana (P50)")
ax2.plot(hist_agg["segundos"], p95_col, color="#d62728", linewidth=2.0, label="Percentil 95 (P95)")
ax2.axhline(200, color="#555555", linestyle=":", linewidth=1.3, label="Referencia SLO (200 ms)")
ax2.set_title("B) Evolución de Latencia P50 y P95 (Escenario Pico)", fontweight="bold")
ax2.set_xlabel("Tiempo transcurrido (segundos)")
ax2.set_ylabel("Tiempo de respuesta (ms)")
ax2.legend(loc="lower right", frameon=True)

# (c) Latencia por endpoint en carga pico (Mediana, P95, P99)
ax3 = axes[1, 0]
ep_ordenado = ep_pico.sort_values("95%", ascending=True)
nombres_cortos = [
    n.replace("POST /predict/", "POST /").replace("GET /", "GET /")
    for n in ep_ordenado["Name"]
]
y_pos = np.arange(len(nombres_cortos))
altura = 0.27

ax3.barh(y_pos - altura, ep_ordenado["50%"], height=altura, color="#2ca02c", label="P50 (Mediana)")
ax3.barh(y_pos, ep_ordenado["95%"], height=altura, color="#ff7f0e", label="P95")
ax3.barh(y_pos + altura, ep_ordenado["99%"], height=altura, color="#d62728", label="P99")
ax3.set_yticks(y_pos)
ax3.set_yticklabels(nombres_cortos, fontsize=9)
ax3.set_title("C) Latencia por Endpoint en Carga Pico (50 usuarios)", fontweight="bold")
ax3.set_xlabel("Tiempo de respuesta (ms)")
ax3.legend(loc="lower right", frameon=True)

# (d) Comparativa agregada Normal vs. Pico
ax4 = axes[1, 1]
metricas_labels = ["RPS\nTotal", "RPS\nInferencia ML", "Latencia\nP50 (ms)", "Latencia\nP95 (ms)"]
vals_normal = [
    float(agg_normal["Requests/s"]),
    rps_inferencia_normal,
    float(agg_normal["50%"]),
    float(agg_normal["95%"]),
]
vals_pico = [
    float(agg_pico["Requests/s"]),
    rps_inferencia_pico,
    float(agg_pico["50%"]),
    float(agg_pico["95%"]),
]
x_idx = np.arange(len(metricas_labels))
ancho = 0.35

barras_n = ax4.bar(x_idx - ancho / 2, vals_normal, width=ancho, color="#4c78a8", label="Normal (10 usuarios)")
barras_p = ax4.bar(x_idx + ancho / 2, vals_pico, width=ancho, color="#e45756", label="Pico (50 usuarios)")
ax4.set_xticks(x_idx)
ax4.set_xticklabels(metricas_labels)
ax4.set_title("D) Comparativa Normal vs. Pico (Tasa de error: 0.00%)", fontweight="bold")
ax4.set_ylabel("Valor (RPS o ms)")
ax4.legend(loc="upper left", frameon=True)
ax4.set_ylim(0, max(vals_pico) * 1.15)

for barra in list(barras_n) + list(barras_p):
    val = barra.get_height()
    ax4.text(
        barra.get_x() + barra.get_width() / 2,
        val + 2.0,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
    )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(FIG_LOCUST, dpi=200)
plt.close()


# ==============================================================================
# 5. GRÁFICA 2: MONITOREO DE RECURSOS (PODMAN STATS) EN CARGA PICO
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
fig.suptitle(
    "Consumo de Recursos de Contenedores (podman stats) durante Prueba Pico — 50 Usuarios",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

colores_cont = {
    "mibici-api": "#1f77b4",
    "mibici-rq-worker": "#ff7f0e",
    "mibici-mlflow": "#9467bd",
    "mibici-redis": "#2ca02c",
}

# (a) CPU % por contenedor
ax_cpu = axes[0]
for nombre, color in colores_cont.items():
    sub = df_stats_pico[df_stats_pico["container_name"] == nombre]
    ax_cpu.plot(sub["elapsed_sec"], sub["cpu_percent"], marker="o", markersize=3, linewidth=2, color=color, label=nombre)
ax_cpu.set_title("A) Consumo de CPU (%) por Contenedor", fontweight="bold")
ax_cpu.set_xlabel("Tiempo transcurrido (segundos)")
ax_cpu.set_ylabel("Uso de CPU (%)")
ax_cpu.legend(loc="upper left", frameon=True)

# (b) Memoria RAM (MB) por contenedor
ax_ram = axes[1]
for nombre, color in colores_cont.items():
    sub = df_stats_pico[df_stats_pico["container_name"] == nombre]
    ax_ram.plot(sub["elapsed_sec"], sub["mem_usage_mb"], marker="o", markersize=3, linewidth=2, color=color, label=nombre)
ax_ram.set_title(
    f"B) Consumo de Memoria RAM (MB) — Pico combinado: {ram_pico_total_mb:.1f} MB",
    fontweight="bold",
)
ax_ram.set_xlabel("Tiempo transcurrido (segundos)")
ax_ram.set_ylabel("Memoria RAM utilizada (MB)")
ax_ram.set_ylim(0, 490)
ax_ram.legend(loc="upper center", bbox_to_anchor=(0.5, 0.86), ncol=2, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(FIG_RECURSOS, dpi=200)
plt.close()


# ==============================================================================
# 6. ANÁLISIS FINOPS: COSTO POR INFERENCIA Y ESCENARIOS DE OPTIMIZACIÓN
# ==============================================================================
# Definimos 3 niveles de volumen mensual de inferencias para reportar la economía
# unitaria como un rango realista (práctica recomendada por FinOps for AI):
#   1. Demanda operativa base MiBici AMG (~360 estaciones x 24 h x 2 modelos P1+P2 x 30.42 días)
#   2. Escenario de carga normal sostenida (10% de utilización del RPS normal observado)
#   3. Capacidad máxima probada en escenario pico (RPS de inferencia ML sostenido 100%)
INF_MES_OPERATIVA_BASE = 360 * 24 * 2 * (HORAS_MES / 24.0)  # ~525,600 inferencias/mes
INF_MES_NORMAL_10PCT = rps_inferencia_normal * 0.10 * SEGUNDOS_MES
INF_MES_CAPACIDAD_PICO = rps_inferencia_pico * SEGUNDOS_MES

resultados_finops = []
for esc in ESCENARIOS_NUBE:
    costo_computo_mes = esc["tarifa_hora_usd"] * esc["horas_mes"]
    costo_total_mes = costo_computo_mes + EBS_COSTO_MES
    costo_total_hora_efectivo = costo_total_mes / esc["horas_mes"]

    # Costo por 1 inferencia y por 1 millón de inferencias en cada nivel de demanda
    costo_inf_base = costo_total_mes / INF_MES_OPERATIVA_BASE
    costo_inf_normal10 = costo_total_mes / INF_MES_NORMAL_10PCT
    costo_inf_cap_pico = costo_total_mes / INF_MES_CAPACIDAD_PICO

    # Costo por hora / por inferencia también escalado al tiempo de operación efectivo del escenario
    inf_cap_en_horas_activas = rps_inferencia_pico * (esc["horas_mes"] * 3600.0)
    costo_inf_cap_activa = costo_total_mes / inf_cap_en_horas_activas

    resultados_finops.append(
        {
            "id": esc["id"],
            "etiqueta": esc["etiqueta"].replace("\n", " "),
            "etiqueta_grafica": esc["etiqueta"],
            "instancia": esc["instancia"],
            "modalidad": esc["modalidad"],
            "tarifa_hora_usd": esc["tarifa_hora_usd"],
            "horas_mes": round(esc["horas_mes"], 1),
            "costo_computo_mes_usd": round(costo_computo_mes, 2),
            "costo_almacenamiento_mes_usd": round(EBS_COSTO_MES, 2),
            "costo_total_mes_usd": round(costo_total_mes, 2),
            "costo_hora_efectivo_usd": round(costo_total_hora_efectivo, 5),
            "costo_por_inferencia_operativa_base_usd": round(costo_inf_base, 8),
            "costo_por_1m_operativa_base_usd": round(costo_inf_base * 1_000_000, 2),
            "costo_por_inferencia_normal10_usd": round(costo_inf_normal10, 8),
            "costo_por_1m_normal10_usd": round(costo_inf_normal10 * 1_000_000, 2),
            "costo_por_inferencia_capacidad_pico_usd": round(costo_inf_cap_activa, 10),
            "costo_por_1m_capacidad_pico_usd": round(costo_inf_cap_activa * 1_000_000, 4),
        }
    )

costo_base_mes = resultados_finops[0]["costo_total_mes_usd"]
for r in resultados_finops:
    ahorro_pct = (1.0 - (r["costo_total_mes_usd"] / costo_base_mes)) * 100.0
    r["ahorro_vs_baseline_pct"] = round(ahorro_pct, 1)


# ==============================================================================
# 7. GRÁFICA 3: ANÁLISIS FINOPS (COSTO MENSUAL Y COSTO POR MILLÓN DE INFERENCIAS)
# ==============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Análisis FinOps — Costo Mensual de Operación y Costo por Millón de Inferencias (USD)",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)

etiquetas_x = [r["etiqueta_grafica"] for r in resultados_finops]
costos_comp = [r["costo_computo_mes_usd"] for r in resultados_finops]
costos_ebs = [r["costo_almacenamiento_mes_usd"] for r in resultados_finops]
costos_tot = [r["costo_total_mes_usd"] for r in resultados_finops]
ahorros = [r["ahorro_vs_baseline_pct"] for r in resultados_finops]
x_f = np.arange(len(etiquetas_x))

# (a) Costo mensual desglosado (Cómputo + EBS 20 GB)
ax_f1 = axes[0]
ax_f1.bar(x_f, costos_comp, color="#3b75af", label="Cómputo EC2 (Linux us-east-1)")
ax_f1.bar(x_f, costos_ebs, bottom=costos_comp, color="#9ecae1", label="Almacenamiento EBS gp3 (20 GB)")
ax_f1.set_xticks(x_f)
ax_f1.set_xticklabels(etiquetas_x, fontsize=8.5)
ax_f1.set_ylabel("Costo mensual total (USD / mes)")
ax_f1.set_title("A) Costo Mensual por Arquitectura y Estrategia FinOps", fontweight="bold")
ax_f1.legend(loc="upper right", frameon=True)

for idx, (tot, ah) in enumerate(zip(costos_tot, ahorros)):
    texto = f"${tot:.2f}" if idx == 0 else f"${tot:.2f}\n(-{ah:.0f}%)"
    ax_f1.text(idx, tot + 0.7, texto, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax_f1.set_ylim(0, max(costos_tot) * 1.22)

# (b) Costo por millón de inferencias en demanda operativa base vs. carga media (10% normal)
ax_f2 = axes[1]
c1m_base = [r["costo_por_1m_operativa_base_usd"] for r in resultados_finops]
c1m_med = [r["costo_por_1m_normal10_usd"] for r in resultados_finops]
ancho_f = 0.36

b1 = ax_f2.bar(
    x_f - ancho_f / 2,
    c1m_base,
    width=ancho_f,
    color="#e45756",
    label=f"Demanda base MiBici (~{INF_MES_OPERATIVA_BASE/1e3:.0f}k inf/mes)",
)
b2 = ax_f2.bar(
    x_f + ancho_f / 2,
    c1m_med,
    width=ancho_f,
    color="#54a24b",
    label=f"Tráfico medio 10% normal (~{INF_MES_NORMAL_10PCT/1e6:.2f}M inf/mes)",
)
ax_f2.set_xticks(x_f)
ax_f2.set_xticklabels(etiquetas_x, fontsize=8.5)
ax_f2.set_ylabel("USD por cada 1,000,000 de inferencias")
ax_f2.set_title("B) Economía Unitaria (Costo por Millón de Inferencias)", fontweight="bold")
ax_f2.legend(loc="upper right", frameon=True)

for barra in list(b1) + list(b2):
    val = barra.get_height()
    ax_f2.text(
        barra.get_x() + barra.get_width() / 2,
        val + 1.0,
        f"${val:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
ax_f2.set_ylim(0, max(c1m_base) * 1.22)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(FIG_FINOPS, dpi=200)
plt.close()


# ==============================================================================
# 8. GUARDAR RESUMEN COMPLETO EN JSON Y MOSTRAR TABLA EN CONSOLA
# ==============================================================================
resumen_final = {
    "team": "Equipo35",
    "locust_normal": {
        "usuarios": 10,
        "duracion_seg": 30,
        "peticiones_totales": int(agg_normal["Request Count"]),
        "fallos_totales": int(agg_normal["Failure Count"]),
        "tasa_error_pct": round(float(agg_normal["Failure Count"]) / float(agg_normal["Request Count"]) * 100.0, 4),
        "rps_total": round(float(agg_normal["Requests/s"]), 2),
        "rps_inferencia_ml": round(rps_inferencia_normal, 2),
        "latencia_p50_ms": float(agg_normal["50%"]),
        "latencia_promedio_ms": round(float(agg_normal["Average Response Time"]), 2),
        "latencia_p95_ms": float(agg_normal["95%"]),
        "latencia_p99_ms": float(agg_normal["99%"]),
        "latencia_max_ms": round(float(agg_normal["Max Response Time"]), 2),
    },
    "locust_pico": {
        "usuarios": 50,
        "duracion_seg": 60,
        "peticiones_totales": int(agg_pico["Request Count"]),
        "fallos_totales": int(agg_pico["Failure Count"]),
        "tasa_error_pct": round(float(agg_pico["Failure Count"]) / float(agg_pico["Request Count"]) * 100.0, 4),
        "rps_total": round(float(agg_pico["Requests/s"]), 2),
        "rps_inferencia_ml": round(rps_inferencia_pico, 2),
        "latencia_p50_ms": float(agg_pico["50%"]),
        "latencia_promedio_ms": round(float(agg_pico["Average Response Time"]), 2),
        "latencia_p95_ms": float(agg_pico["95%"]),
        "latencia_p99_ms": float(agg_pico["99%"]),
        "latencia_max_ms": round(float(agg_pico["Max Response Time"]), 2),
        "por_endpoint": [
            {
                "endpoint": str(fila["Name"]),
                "peticiones": int(fila["Request Count"]),
                "fallos": int(fila["Failure Count"]),
                "rps": round(float(fila["Requests/s"]), 2),
                "p50_ms": float(fila["50%"]),
                "promedio_ms": round(float(fila["Average Response Time"]), 2),
                "p95_ms": float(fila["95%"]),
                "p99_ms": float(fila["99%"]),
                "max_ms": round(float(fila["Max Response Time"]), 2),
            }
            for _, fila in ep_pico.iterrows()
        ],
    },
    "recursos_podman_stats": {
        "normal_10_usuarios": recursos_normal,
        "pico_50_usuarios": recursos_pico,
        "totales_pico_4_servicios": {
            "cpu_promedio_combinada_pct": cpu_prom_total_pct,
            "cpu_pico_combinada_pct": cpu_pico_total_pct,
            "ram_promedio_combinada_mb": ram_prom_total_mb,
            "ram_pico_combinada_mb": ram_pico_total_mb,
            "ram_pico_solo_inferencia_sin_mlflow_mb": ram_inferencia_pico_mb,
        },
        "optimizacion_mlflow_aplicada": {
            "ram_antes_mb": 2041.0,
            "ram_despues_mb": recursos_pico["mibici-mlflow"]["ram_pico_mb"],
            "ahorro_ram_mb": round(2041.0 - recursos_pico["mibici-mlflow"]["ram_pico_mb"], 1),
            "ahorro_ram_pct": round((1.0 - recursos_pico["mibici-mlflow"]["ram_pico_mb"] / 2041.0) * 100.0, 1),
        },
    },
    "finops": {
        "region_referencia": "AWS us-east-1 (Linux)",
        "horas_mes_estandar": HORAS_MES,
        "almacenamiento_ebs_gp3_gb": EBS_GB,
        "almacenamiento_ebs_mes_usd": EBS_COSTO_MES,
        "volumenes_mensuales_referencia": {
            "operativa_base_mibici_360_estaciones": round(INF_MES_OPERATIVA_BASE, 0),
            "trafico_medio_10pct_normal": round(INF_MES_NORMAL_10PCT, 0),
            "capacidad_maxima_pico_100pct": round(INF_MES_CAPACIDAD_PICO, 0),
        },
        "escenarios": resultados_finops,
    },
}

with open(JSON_FINOPS, "w", encoding="utf-8") as f_json:
    json.dump(resumen_final, f_json, indent=4, ensure_ascii=False)

print(f"\nGráficas generadas correctamente:")
print(f"  1. {FIG_LOCUST.relative_to(PROJ_ROOT)}")
print(f"  2. {FIG_RECURSOS.relative_to(PROJ_ROOT)}")
print(f"  3. {FIG_FINOPS.relative_to(PROJ_ROOT)}")
print(f"  4. {JSON_FINOPS.relative_to(PROJ_ROOT)}")

print("\n--- TABLA RESUMEN FINOPS (AWS us-east-1 Linux + 20 GB EBS gp3) ---")
for r in resultados_finops:
    print(
        f"  {r['etiqueta']:<36} | ${r['costo_total_mes_usd']:5.2f}/mes "
        f"(ahorro {r['ahorro_vs_baseline_pct']:4.1f}%) | "
        f"Base: ${r['costo_por_1m_operativa_base_usd']:5.2f}/1M inf | "
        f"Medio: ${r['costo_por_1m_normal10_usd']:5.2f}/1M inf | "
        f"Pico cap: ${r['costo_por_1m_capacidad_pico_usd']:.4f}/1M inf"
    )
print("=" * 78)