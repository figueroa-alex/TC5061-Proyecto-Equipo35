"""
Módulo de recuperación y exportación de modelos desde el Model Registry de MLflow (Equipo 35).

Se implementa una estrategia híbrida reproducible:
  1. Recuperación directa desde el Model Registry de MLflow (configurado mediante la
     variable de entorno MLFLOW_TRACKING_URI) utilizando el nombre oficial del modelo
     y su alias (@champion) o versión numérica (v2).
  2. Exportación local de los modelos registrados hacia 'models/registry_bundle/' para
     empaquetarlos dentro de la imagen del contenedor Podman, garantizando que la API
     funcione en cualquier máquina incluso si el servidor externo de MLflow está apagado.

Cómo exportar los modelos del Registry antes de construir la imagen:
    python -m mibici.api.model_loader
"""

import json
from pathlib import Path
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from mibici import config

# ==============================================================================
# MAPA CENTRALIZADO DE LOS TRES MODELOS CAMPEONES DEL PROYECTO MIBICI
# ==============================================================================
MAPA_MODELOS_MIBICI = {
    "p1": {
        "nombre_registro": config.MODEL_NAME_P1,
        "carpeta_bundle": config.BUNDLE_DIR_P1,
        "tarea": "regression",
        "descripcion": "Predicción de viajes iniciados por estación en la siguiente hora (P1).",
    },
    "p2": {
        "nombre_registro": config.MODEL_NAME_P2,
        "carpeta_bundle": config.BUNDLE_DIR_P2,
        "tarea": "classification",
        "descripcion": "Predicción de riesgo de desbalance horario >= 30% de capacidad (P2).",
    },
    "p3": {
        "nombre_registro": config.MODEL_NAME_P3,
        "carpeta_bundle": config.BUNDLE_DIR_P3,
        "tarea": "clustering",
        "descripcion": "Asignación de grupo operativo y geográfico de estación en el AMG (P3).",
    },
}


# ==============================================================================
# FUNCIÓN 1: CONSTRUIR LAS URIS DEL MODEL REGISTRY POR ALIAS Y POR VERSIÓN
# ==============================================================================
def obtener_uris_model_registry(nombre_modelo: str) -> tuple[str, str]:
    """
    Construye las dos URIs oficiales de MLflow Model Registry para un modelo:
      - URI por alias:   models:/<nombre_modelo>@champion
      - URI por versión: models:/<nombre_modelo>/<version>
    """
    alias = config.MODEL_REGISTRY_ALIAS
    version = config.MODEL_REGISTRY_VERSION
    uri_por_alias = f"models:/{nombre_modelo}@{alias}"
    uri_por_version = f"models:/{nombre_modelo}/{version}"
    return uri_por_alias, uri_por_version


# ==============================================================================
# FUNCIÓN 2: EXPORTAR LOS MODELOS DESDE EL MODEL REGISTRY PARA EL CONTENEDOR
# ==============================================================================
def exportar_modelos_desde_registry() -> dict:
    """
    Se conecta al servidor de MLflow (MLFLOW_TRACKING_URI), recupera cada uno de los
    tres modelos desde el Model Registry usando su nombre y alias (@champion) o versión,
    descarga sus artefactos completos a 'models/registry_bundle/' y guarda un manifiesto
    JSON con el linaje exacto (versión, alias, run_id, commit y versión de datos).
    """
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cliente = MlflowClient()
    config.REGISTRY_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    manifiesto = {
        "tracking_uri_origen": config.MLFLOW_TRACKING_URI,
        "team": config.TEAM_TAG,
        "models": {},
    }

    print("=" * 75)
    print("EXPORTANDO MODELOS DESDE EL MODEL REGISTRY DE MLFLOW PARA EL CONTENEDOR")
    print("=" * 75)

    for clave_pregunta, info in MAPA_MODELOS_MIBICI.items():
        nombre_registro = info["nombre_registro"]
        carpeta_destino: Path = info["carpeta_bundle"]
        uri_alias, uri_version = obtener_uris_model_registry(nombre_registro)

        # 1. Consultamos en el Model Registry los metadatos de la versión apuntada por @champion
        try:
            version_info = cliente.get_model_version_by_alias(
                name=nombre_registro,
                alias=config.MODEL_REGISTRY_ALIAS,
            )
            uri_seleccionada = uri_alias
        except Exception:
            version_info = cliente.get_model_version(
                name=nombre_registro,
                version=config.MODEL_REGISTRY_VERSION,
            )
            uri_seleccionada = uri_version

        # 2. Limpiamos la carpeta destino si existía una exportación previa
        if carpeta_destino.exists():
            shutil.rmtree(carpeta_destino)
        carpeta_destino.mkdir(parents=True, exist_ok=True)

        # 3. Descargamos el modelo directamente desde el Model Registry
        mlflow.artifacts.download_artifacts(
            artifact_uri=uri_seleccionada,
            dst_path=str(carpeta_destino),
        )

        tags_version = version_info.tags if hasattr(version_info, "tags") else {}
        metadatos_modelo = {
            "question": clave_pregunta,
            "model_name": nombre_registro,
            "model_version": str(version_info.version),
            "model_alias": config.MODEL_REGISTRY_ALIAS,
            "registry_uri_alias": uri_alias,
            "registry_uri_version": f"models:/{nombre_registro}/{version_info.version}",
            "run_id": version_info.run_id,
            "git_commit": tags_version.get("git_commit", "9f8eea9"),
            "data_version": tags_version.get("data_version", "128979137b46"),
            "bundle_relative_path": str(carpeta_destino.relative_to(config.PROJ_ROOT)),
        }
        manifiesto["models"][clave_pregunta] = metadatos_modelo

        print(f"  -> [{clave_pregunta.upper()}] {nombre_registro}")
        print(f"     URI recuperada:  {uri_seleccionada} (Versión v{version_info.version})")
        print(f"     Run ID origen:   {version_info.run_id}")
        print(f"     Empaquetado en:  {carpeta_destino.relative_to(config.PROJ_ROOT)}")

    with open(config.BUNDLE_MANIFEST_FILE, "w", encoding="utf-8") as archivo_json:
        json.dump(manifiesto, archivo_json, indent=4, ensure_ascii=False)

    print("-" * 75)
    print(f"Manifiesto guardado en: {config.BUNDLE_MANIFEST_FILE.relative_to(config.PROJ_ROOT)}")
    print("=" * 75)
    return manifiesto


# ==============================================================================
# FUNCIÓN 3: LEER METADATOS DEL MANIFIESTO LOCAL SI EXISTE
# ==============================================================================
def leer_metadatos_bundle(pregunta: str) -> dict:
    """
    Lee del archivo registry_manifest.json los metadatos del modelo exportado
    (nombre, versión, alias y versión de datos) para incluirlos en las respuestas.
    """
    info_base = MAPA_MODELOS_MIBICI[pregunta]
    nombre_registro = info_base["nombre_registro"]
    uri_alias, uri_version = obtener_uris_model_registry(nombre_registro)

    metadatos_por_defecto = {
        "question": pregunta,
        "model_name": nombre_registro,
        "model_version": str(config.MODEL_REGISTRY_VERSION),
        "model_alias": config.MODEL_REGISTRY_ALIAS,
        "registry_uri_alias": uri_alias,
        "registry_uri_version": uri_version,
        "data_version": "128979137b46",
    }

    if config.BUNDLE_MANIFEST_FILE.exists():
        with open(config.BUNDLE_MANIFEST_FILE, "r", encoding="utf-8") as f_json:
            contenido = json.load(f_json)
            datos_pregunta = contenido.get("models", {}).get(pregunta)
            if datos_pregunta:
                metadatos_por_defecto.update(datos_pregunta)

    return metadatos_por_defecto


# ==============================================================================
# FUNCIÓN 4: CARGAR UN MODELO EN LA API CON ESTRATEGIA HÍBRIDA
# ==============================================================================
def cargar_modelo_para_api(pregunta: str, modo_carga: str | None = None) -> tuple[object, dict]:
    """
    Carga y devuelve el modelo de la pregunta indicada ('p1', 'p2' o 'p3') junto con
    su diccionario de metadatos, aplicando la estrategia configurada:
      - Paso A: Si el modo es 'hybrid' o 'registry', intenta recuperar el modelo en vivo
                desde el servidor MLFLOW_TRACKING_URI usando su URI por alias (@champion)
                y, como segunda opción, su URI por versión (models:/<nombre>/<version>).
      - Paso B: Si el servidor MLflow no está disponible o el modo es 'bundle', carga el
                modelo descargado del Registry y empaquetado en 'models/registry_bundle/'.
    """
    clave = pregunta.lower().strip()
    if clave not in MAPA_MODELOS_MIBICI:
        raise ValueError(f"Pregunta no válida: '{pregunta}'. Usa 'p1', 'p2' o 'p3'.")

    modo = (modo_carga or config.MODEL_LOAD_MODE).lower().strip()
    info = MAPA_MODELOS_MIBICI[clave]
    nombre_registro = info["nombre_registro"]
    carpeta_bundle: Path = info["carpeta_bundle"]
    uri_alias, uri_version = obtener_uris_model_registry(nombre_registro)
    metadatos = leer_metadatos_bundle(clave)

    # --------------------------------------------------------------------------
    # OPCIÓN 1: Intentar cargar en vivo desde el servidor MLflow Model Registry
    # --------------------------------------------------------------------------
    if modo in ("hybrid", "registry"):
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        for uri_intento in (uri_alias, uri_version):
            try:
                modelo_remoto = mlflow.pyfunc.load_model(uri_intento)
                metadatos["loaded_from"] = f"mlflow_registry ({config.MLFLOW_TRACKING_URI})"
                metadatos["active_uri"] = uri_intento
                return modelo_remoto, metadatos
            except Exception as error_remoto:
                ultimo_error = error_remoto

        if modo == "registry":
            raise RuntimeError(
                f"No fue posible recuperar '{nombre_registro}' desde el Model Registry "
                f"en {config.MLFLOW_TRACKING_URI}: {ultimo_error}"
            )

    # --------------------------------------------------------------------------
    # OPCIÓN 2: Cargar desde el paquete del Model Registry incluido en la imagen
    # --------------------------------------------------------------------------
    archivo_mlmodel = carpeta_bundle / "MLmodel"
    if archivo_mlmodel.exists():
        modelo_local = mlflow.pyfunc.load_model(str(carpeta_bundle))
        metadatos["loaded_from"] = "packaged_registry_bundle (models/registry_bundle)"
        metadatos["active_uri"] = metadatos.get("registry_uri_alias", uri_alias)
        return modelo_local, metadatos

    raise FileNotFoundError(
        f"No se encontró el modelo '{nombre_registro}' ni en el servidor MLflow "
        f"({config.MLFLOW_TRACKING_URI}) ni en '{carpeta_bundle}'. "
        "Ejecuta primero: python -m mibici.api.model_loader"
    )


def cargar_todos_los_modelos(modo_carga: str | None = None) -> dict[str, dict]:
    """
    Carga los tres modelos del proyecto MiBici (P1, P2 y P3) al iniciar la API
    y los devuelve en un diccionario listo para responder peticiones HTTP.
    """
    catalogo_cargado = {}
    for clave_pregunta in ("p1", "p2", "p3"):
        modelo, metadatos = cargar_modelo_para_api(clave_pregunta, modo_carga=modo_carga)
        catalogo_cargado[clave_pregunta] = {
            "model": modelo,
            "metadata": metadatos,
        }
    return catalogo_cargado


if __name__ == "__main__":
    exportar_modelos_desde_registry()