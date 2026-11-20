"""
Monitor lineal de consumo de CPU y memoria de contenedores usando podman/docker stats (Equipo 35).

Ejecuta 'podman stats --no-stream --format json' (o docker) cada N segundos durante
una prueba de carga con Locust, guarda las mediciones en un archivo CSV y muestra
al final el consumo promedio y pico por contenedor.

Uso:
    python load_tests/monitorear_recursos.py --engine podman --duration 65 --interval 2 \
        --output reports/operations/podman_stats_pico.csv
"""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import time

# Contenedores del proyecto MiBici que deseamos monitorear
CONTENEDORES_MIBICI = {
    "mibici-api",
    "mibici-rq-worker",
    "mibici-redis",
    "mibici-mlflow",
}


def convertir_a_mb(texto_tamano: str) -> float:
    """Convierte una cadena de memoria (ej. '241.1MB', '1.074GB', '4264kB') a megabytes (MB)."""
    if not texto_tamano:
        return 0.0
    coincidencia = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)", texto_tamano.strip())
    if not coincidencia:
        return 0.0

    valor = float(coincidencia.group(1))
    unidad = coincidencia.group(2).lower()

    if unidad in ("b", "bytes"):
        return valor / (1000.0 * 1000.0)
    if unidad in ("kb", "kib"):
        return valor / 1000.0
    if unidad in ("mb", "mib"):
        return valor
    if unidad in ("gb", "gib"):
        return valor * 1000.0
    return valor


def extraer_porcentaje(texto_pct: str) -> float:
    """Extrae un número flotante de una cadena con porcentaje (ej. '14.97%' -> 14.97)."""
    if not texto_pct:
        return 0.0
    limpio = str(texto_pct).replace("%", "").strip()
    try:
        return float(limpio)
    except ValueError:
        return 0.0


def consultar_muestra_stats(engine: str) -> list[dict]:
    """Ejecuta '<engine> stats --no-stream --format json' y devuelve una lista de registros."""
    comando = [engine, "stats", "--no-stream", "--format", "json"]
    proceso = subprocess.run(
        comando,
        capture_output=True,
        text=True,
        check=False,
    )
    salida = proceso.stdout.strip()
    if proceso.returncode != 0 or not salida:
        return []

    # Podman devuelve un arreglo JSON; Docker devuelve un objeto JSON por línea
    if salida.startswith("["):
        crudos = json.loads(salida)
    else:
        crudos = [json.loads(linea) for linea in salida.splitlines() if linea.strip()]

    registros = []
    for item in crudos:
        nombre = str(item.get("name") or item.get("Name") or "").strip()
        if nombre not in CONTENEDORES_MIBICI:
            continue

        cpu_pct = extraer_porcentaje(item.get("cpu_percent") or item.get("CPUPerc") or "0")
        mem_pct = extraer_porcentaje(item.get("mem_percent") or item.get("MemPerc") or "0")

        # Campo típico: "241.1MB / 1.611GB"
        mem_bruta = str(item.get("mem_usage") or item.get("MemUsage") or "0MB / 0MB")
        partes_mem = mem_bruta.split("/")
        mem_usada_mb = convertir_a_mb(partes_mem[0]) if len(partes_mem) >= 1 else 0.0
        mem_limite_mb = convertir_a_mb(partes_mem[1]) if len(partes_mem) >= 2 else 0.0

        pids_bruto = str(item.get("pids") or item.get("PIDs") or "0").strip()
        pids = int(pids_bruto) if pids_bruto.isdigit() else 0

        registros.append(
            {
                "container_name": nombre,
                "cpu_percent": round(cpu_pct, 2),
                "mem_usage_mb": round(mem_usada_mb, 2),
                "mem_limit_mb": round(mem_limite_mb, 2),
                "mem_percent": round(mem_pct, 2),
                "pids": pids,
            }
        )
    return registros


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Registra CPU y memoria de los contenedores MiBici durante pruebas de carga."
    )
    parser.add_argument("--engine", default="podman", choices=["podman", "docker"])
    parser.add_argument("--duration", type=int, default=35, help="Duración total en segundos.")
    parser.add_argument("--interval", type=int, default=2, help="Intervalo entre muestras (s).")
    parser.add_argument(
        "--output",
        default="reports/operations/podman_stats_carga.csv",
        help="Ruta del archivo CSV de salida.",
    )
    args = parser.parse_args()

    ruta_salida = Path(args.output)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    columnas = [
        "timestamp_utc",
        "elapsed_sec",
        "container_name",
        "cpu_percent",
        "mem_usage_mb",
        "mem_limit_mb",
        "mem_percent",
        "pids",
    ]

    inicio = time.monotonic()
    filas_totales = []

    print(
        f"[Monitor] Iniciando captura con '{args.engine} stats' por {args.duration}s "
        f"(cada {args.interval}s) -> {ruta_salida}"
    )

    while True:
        transcurrido = round(time.monotonic() - inicio, 1)
        if transcurrido > args.duration:
            break

        marca_tiempo = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        muestra = consultar_muestra_stats(args.engine)
        for fila in muestra:
            filas_totales.append(
                {
                    "timestamp_utc": marca_tiempo,
                    "elapsed_sec": transcurrido,
                    **fila,
                }
            )

        restante = args.duration - (time.monotonic() - inicio)
        if restante <= 0:
            break
        time.sleep(min(args.interval, restante))

    with open(ruta_salida, "w", newline="", encoding="utf-8") as archivo_csv:
        escritor = csv.DictWriter(archivo_csv, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas_totales)

    print(f"[Monitor] Captura finalizada: {len(filas_totales)} mediciones guardadas en {ruta_salida}")

    # Imprimir resumen promedio y máximo por contenedor
    print("\n--- Resumen de consumo de recursos observado ---")
    for nombre in sorted(CONTENEDORES_MIBICI):
        sub = [f for f in filas_totales if f["container_name"] == nombre]
        if not sub:
            continue
        cpu_prom = sum(f["cpu_percent"] for f in sub) / len(sub)
        cpu_max = max(f["cpu_percent"] for f in sub)
        mem_prom = sum(f["mem_usage_mb"] for f in sub) / len(sub)
        mem_max = max(f["mem_usage_mb"] for f in sub)
        print(
            f"  {nombre:<18} | CPU prom: {cpu_prom:6.2f}% | CPU pico: {cpu_max:6.2f}% "
            f"| RAM prom: {mem_prom:6.1f} MB | RAM pico: {mem_max:6.1f} MB"
        )


if __name__ == "__main__":
    main()