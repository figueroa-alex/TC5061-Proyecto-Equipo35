# ==============================================================================
# CONTAINERFILE (ESTÁNDAR OCI) - MICROSERVICIOS PROYECTO MIBICI (EQUIPO 35)
# Compatible con Podman (podman build / podman-compose) y Docker (docker compose)
# Reutilizado por: mibici-api, rq-worker y mlflow-server
# ==============================================================================

# 1. Imagen base oficial de Python con la misma versión del proyecto (3.14.3)
FROM docker.io/library/python:3.14.3-slim

# 2. Variables de entorno por defecto para optimizar Python y configurar los servicios
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_TRACKING_URI="http://mlflow-server:5000" \
    MLFLOW_EXPERIMENT_NAME="mibici-equipo35" \
    MIBICI_MODEL_ALIAS="champion" \
    MIBICI_MODEL_VERSION="3" \
    MIBICI_MODEL_LOAD_MODE="hybrid" \
    MIBICI_API_HOST="0.0.0.0" \
    MIBICI_API_PORT="8000" \
    REDIS_URL="redis://redis:6379/0" \
    RQ_QUEUE_NAME="mibici_predicciones"

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalar librería del sistema libgomp1 (requerida por Scikit-Learn HistGradientBoosting)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar e instalar dependencias de Python (excluyendo editable local y paquete solo-macOS)
COPY requirements.txt /app/requirements.txt
RUN grep -Ev "^(-e |appnope==)" /app/requirements.txt > /tmp/requirements_docker.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements_docker.txt \
    && rm /tmp/requirements_docker.txt

# 6. Copiar metadatos del paquete, código fuente y modelos campeones del Model Registry
COPY pyproject.toml README.md LICENSE .env.example /app/
COPY mibici /app/mibici
COPY models/registry_bundle /app/models/registry_bundle
COPY data/processed/data_version.json /app/data/processed/data_version.json

# 7. Instalar el paquete local 'mibici' sin reinstalar dependencias
RUN pip install --no-deps -e .

# 8. Crear usuario sin privilegios y carpeta /mlflow con permisos para el volumen persistente
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /mlflow/mlartifacts \
    && chown -R appuser:appuser /app /mlflow
USER appuser

# 9. Documentar el puerto principal expuesto por FastAPI / Uvicorn
EXPOSE 8000

# 10. Comando de arranque por defecto (servicio API REST)
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]