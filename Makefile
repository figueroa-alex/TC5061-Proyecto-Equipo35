# ==============================================================================
# CONTAINERFILE (ESTÁNDAR OCI) - SERVICIO API REST PROYECTO MIBICI (EQUIPO 35)
# Compatible con Podman (podman build) y Docker (docker build)
# ==============================================================================

# 1. Imagen base oficial de Python con la misma versión del proyecto (3.14.3)
FROM docker.io/library/python:3.14.3-slim

# 2. Variables de entorno para optimizar Python y configurar la API dentro del contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # Cuando el servidor MLflow corre en la máquina host, el contenedor se conecta a:
    MLFLOW_TRACKING_URI="http://host.containers.internal:5000" \
    MLFLOW_EXPERIMENT_NAME="mibici-equipo35" \
    MIBICI_MODEL_ALIAS="champion" \
    MIBICI_MODEL_VERSION="2" \
    # Estrategia híbrida: intenta MLFLOW_TRACKING_URI y usa models/registry_bundle como respaldo
    MIBICI_MODEL_LOAD_MODE="hybrid" \
    MIBICI_API_HOST="0.0.0.0" \
    MIBICI_API_PORT="8000"

# 3. Directorio de trabajo dentro del contenedor
WORKDIR /app

# 4. Instalar librería del sistema libgomp1 (requerida por Scikit-Learn HistGradientBoosting)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 5. Copiar e instalar las dependencias de Python (aprovechando el caché de capas OCI)
COPY requirements.txt /app/requirements.txt
RUN grep -v "^-e " /app/requirements.txt > /tmp/requirements_docker.txt \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements_docker.txt \
    && rm /tmp/requirements_docker.txt

# 6. Copiar metadatos del paquete, código fuente y modelos descargados del Model Registry
COPY pyproject.toml README.md LICENSE .env.example /app/
COPY mibici /app/mibici
COPY models/registry_bundle /app/models/registry_bundle
COPY data/processed/data_version.json /app/data/processed/data_version.json

# 7. Instalar el paquete local 'mibici' sin reinstalar dependencias
RUN pip install --no-deps -e .

# 8. Crear un usuario sin privilegios de superusuario (buena práctica de seguridad en contenedores)
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# 9. Documentar el puerto expuesto por FastAPI / Uvicorn
EXPOSE 8000

# 10. Comando de arranque del servidor de inferencia REST
CMD ["uvicorn", "mibici.api.main:app", "--host", "0.0.0.0", "--port", "8000"]