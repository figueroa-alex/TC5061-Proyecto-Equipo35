"""
Script de gestión del Model Registry, asignación de aliases (@champion / @challenger),
verificación de inferencia en producción y resumen de experimentos (Equipo 35).

Cómo ejecutarlo con MLflow Project:
  - Para registrar los mejores modelos y verificar su carga desde @champion:
      mlflow run . -e register --env-manager=local -P action=register_and_verify
  - Para imprimir en consola la tabla comparativa de todas las corridas del experimento:
      mlflow run . -e register --env-manager=local -P action=summary
"""

import argparse
import os

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

from mibici import config


# ==============================================================================
# FUNCIÓN AUXILIAR: LIMPIAR CORRIDAS FALLIDAS DE PRUEBAS PREVIAS
# ==============================================================================
def limpiar_corridas_fallidas(cliente: MlflowClient, experimento_id: str) -> None:
    """
    Elimina del experimento cualquier corrida que haya quedado en estado FAILED
    durante pruebas anteriores para mantener limpia la interfaz de MLflow.
    """
    corridas_fallidas = cliente.search_runs(
        experiment_ids=[experimento_id],
        filter_string="attributes.status = 'FAILED'",
    )
    for run_fallida in corridas_fallidas:
        cliente.delete_run(run_fallida.info.run_id)


# ==============================================================================
# FUNCIÓN 1: IMPRIMIR RESUMEN DE TODAS LAS CORRIDAS DEL EXPERIMENTO PARA EL REPORTE
# ==============================================================================
def imprimir_resumen_experimentos(cliente: MlflowClient, experimento_id: str) -> None:
    """
    Consulta todas las corridas finalizadas con éxito en el experimento mibici-equipo35
    y las imprime ordenadas por pregunta para facilitar la elaboración del reporte.
    """
    corridas = cliente.search_runs(
        experiment_ids=[experimento_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time ASC"],
    )

    print("\n" + "=" * 95)
    print(f"TABLA RESUMEN DE CORRIDAS EN EXPERIMENTO '{config.MLFLOW_EXPERIMENT_NAME}'")
    print("=" * 95)

    for pregunta, metrica_principal in [
        ("p1", "test_mae"),
        ("p2", "test_f1"),
        ("p3", "train_silhouette"),
    ]:
        print(f"\n--- PREGUNTA: {pregunta.upper()} (Métrica clave: {metrica_principal}) ---")
        for run in corridas:
            tags = run.data.tags
            if tags.get("question") != pregunta:
                continue
            nombre = tags.get("mlflow.runName", run.info.run_name)
            etapa = tags.get("stage_of_work", "N/A")
            familia = tags.get("model_family", "N/A")
            commit = tags.get("git_commit", "N/A")
            version_datos = tags.get("data_version", "N/A")
            es_hija = " (Nested Hija)" if "mlflow.parentRunId" in tags else ""
            m = run.data.metrics
            p = run.data.params

            if pregunta == "p1":
                print(
                    f"  * {nombre}{es_hija} | run_id={run.info.run_id[:8]} | stage={etapa} | "
                    f"family={familia} | val_mae={m.get('val_mae', m.get('best_child_val_mae', 0.0)):.4f} | "
                    f"test_mae={m.get('test_mae', 0.0):.4f} | test_rmse={m.get('test_rmse', 0.0):.4f} | "
                    f"test_r2={m.get('test_r2', 0.0):.4f} | mejora_pct={m.get('improvement_over_baseline_pct', 0.0):.2f}% | "
                    f"time={m.get('training_time_sec', 0.0):.2f}s | params={p}"
                )
            elif pregunta == "p2":
                print(
                    f"  * {nombre}{es_hija} | run_id={run.info.run_id[:8]} | stage={etapa} | "
                    f"family={familia} | val_f1={m.get('val_f1', m.get('best_child_val_f1', 0.0)):.4f} | "
                    f"test_f1={m.get('test_f1', 0.0):.4f} | test_recall={m.get('test_recall', 0.0):.4f} | "
                    f"test_precision={m.get('test_precision', 0.0):.4f} | test_roc_auc={m.get('test_roc_auc', 0.0):.4f} | "
                    f"test_pr_auc={m.get('test_pr_auc', 0.0):.4f} | time={m.get('training_time_sec', 0.0):.2f}s | params={p}"
                )
            elif pregunta == "p3":
                print(
                    f"  * {nombre}{es_hija} | run_id={run.info.run_id[:8]} | stage={etapa} | "
                    f"family={familia} | silhouette={m.get('train_silhouette', m.get('best_child_silhouette', 0.0)):.4f} | "
                    f"davies_bouldin={m.get('train_davies_bouldin', 0.0):.4f} | "
                    f"min_cluster_share={m.get('min_cluster_share', 0.0):.4f} | "
                    f"commit={commit} | data_ver={version_datos} | params={p}"
                )
    print("=" * 95)


# ==============================================================================
# FUNCIÓN 2: REGISTRAR MODELO EN MODEL REGISTRY CON LINAJE Y ALIASES
# ==============================================================================
def registrar_campeon_y_retador(
    cliente: MlflowClient,
    experimento_id: str,
    pregunta: str,
    nombre_registro: str,
    descripcion_modelo: str,
    orden_metrica: str,
) -> None:
    """
    Busca las corridas candidatas ('stage_of_work' = 'candidate') finalizadas con éxito,
    las ordena según la métrica de validación/entrenamiento, registra la mejor
    como @champion y la segunda mejor como @challenger en el Model Registry,
    conservando todo su linaje (experimento, corrida, commit y versión de datos).
    """
    filtro = (
        f"tags.question = '{pregunta}' and "
        f"tags.stage_of_work = 'candidate' and "
        f"attributes.status = 'FINISHED'"
    )
    corridas_candidatas = cliente.search_runs(
        experiment_ids=[experimento_id],
        filter_string=filtro,
        order_by=[orden_metrica],
    )

    if len(corridas_candidatas) == 0:
        raise RuntimeError(
            f"No se encontraron corridas candidatas finalizadas para {pregunta}. "
            f"Ejecuta primero los entrenamientos con 'mlflow run . -e train'."
        )

    corrida_champion = corridas_candidatas[0]
    corrida_challenger = (
        corridas_candidatas[1] if len(corridas_candidatas) > 1 else corridas_candidatas[0]
    )

    print(f"\n--- Registrando modelos para {pregunta.upper()} en '{nombre_registro}' ---")

    # 1. Aseguramos que el modelo registrado exista y tenga su descripción general
    try:
        cliente.create_registered_model(
            name=nombre_registro,
            description=descripcion_modelo,
            tags={"team": config.TEAM_TAG, "question": pregunta},
        )
    except Exception:
        cliente.update_registered_model(
            name=nombre_registro,
            description=descripcion_modelo,
        )

    # 2. Registramos la versión Challenger (segunda mejor candidata) si existe más de una
    if len(corridas_candidatas) > 1:
        tags_ch = corrida_challenger.data.tags
        uri_challenger = tags_ch.get(
            "logged_model_uri", f"runs:/{corrida_challenger.info.run_id}/model"
        )
        ver_ch = mlflow.register_model(model_uri=uri_challenger, name=nombre_registro)
        desc_version_ch = (
            f"Modelo Challenger ({tags_ch.get('mlflow.runName', '')}). "
            f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} (ID {experimento_id}) | "
            f"Run ID: {corrida_challenger.info.run_id} | "
            f"Git Commit: {tags_ch.get('git_commit', 'N/A')} | "
            f"Data Version: {tags_ch.get('data_version', 'N/A')}."
        )
        cliente.update_model_version(
            name=nombre_registro,
            version=ver_ch.version,
            description=desc_version_ch,
        )
        cliente.set_model_version_tag(
            name=nombre_registro,
            version=ver_ch.version,
            key="git_commit",
            value=tags_ch.get("git_commit", "N/A"),
        )
        cliente.set_model_version_tag(
            name=nombre_registro,
            version=ver_ch.version,
            key="data_version",
            value=tags_ch.get("data_version", "N/A"),
        )
        cliente.set_registered_model_alias(
            name=nombre_registro,
            alias=config.CHALLENGER_ALIAS,
            version=ver_ch.version,
        )
        print(
            f"  [Challenger] Versión v{ver_ch.version} asignada al alias '@{config.CHALLENGER_ALIAS}' "
            f"(Run: {tags_ch.get('mlflow.runName')} | ID: {corrida_challenger.info.run_id})"
        )

    # 3. Registramos la versión Champion (mejor candidata, pasa a producción)
    tags_cp = corrida_champion.data.tags
    uri_champion = tags_cp.get(
        "logged_model_uri", f"runs:/{corrida_champion.info.run_id}/model"
    )
    ver_cp = mlflow.register_model(model_uri=uri_champion, name=nombre_registro)
    desc_version_cp = (
        f"Modelo Champion en Producción ({tags_cp.get('mlflow.runName', '')}). "
        f"Linaje -> Experimento: {config.MLFLOW_EXPERIMENT_NAME} (ID {experimento_id}) | "
        f"Run ID: {corrida_champion.info.run_id} | "
        f"Git Commit: {tags_cp.get('git_commit', 'N/A')} | "
        f"Data Version: {tags_cp.get('data_version', 'N/A')}."
    )
    cliente.update_model_version(
        name=nombre_registro,
        version=ver_cp.version,
        description=desc_version_cp,
    )
    cliente.set_model_version_tag(
        name=nombre_registro,
        version=ver_cp.version,
        key="git_commit",
        value=tags_cp.get("git_commit", "N/A"),
    )
    cliente.set_model_version_tag(
        name=nombre_registro,
        version=ver_cp.version,
        key="data_version",
        value=tags_cp.get("data_version", "N/A"),
    )
    cliente.set_registered_model_alias(
        name=nombre_registro,
        alias=config.PRODUCTION_ALIAS,
        version=ver_cp.version,
    )
    print(
        f"  [Champion]   Versión v{ver_cp.version} asignada al alias '@{config.PRODUCTION_ALIAS}' "
        f"(Run: {tags_cp.get('mlflow.runName')} | ID: {corrida_champion.info.run_id} | "
        f"Commit: {tags_cp.get('git_commit')} | Data: {tags_cp.get('data_version')})"
    )


# ==============================================================================
# FUNCIÓN 3: VERIFICAR QUE LOS 3 MODELOS EN PRODUCCIÓN SE CARGAN POR ALIAS
# ==============================================================================
def verificar_carga_modelos_produccion() -> None:
    """
    Cumple el Paso 8 de la Parte C: carga cada modelo en estado de producción
    usando su URI 'models:/<nombre>@champion' y realiza una inferencia de prueba.
    """
    print("\n" + "=" * 75)
    print("VERIFICACIÓN DE CARGA E INFERENCIA DESDE EL MODEL REGISTRY (@champion)")
    print("=" * 75)

    df_p1_p2 = pd.read_parquet(config.INTERIM_P1_P2_FILE)
    columnas_x_p1_p2 = [
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
    muestra_p1_p2 = df_p1_p2[columnas_x_p1_p2].tail(3).astype("float64")

    df_p3 = pd.read_parquet(config.INTERIM_P3_FILE)
    columnas_x_p3 = [
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
    muestra_p3 = df_p3[columnas_x_p3].head(3).astype("float64")

    for pregunta, nombre_modelo, muestra in [
        ("P1 (Demanda Horaria)", config.MODEL_NAME_P1, muestra_p1_p2),
        ("P2 (Riesgo de Desbalance)", config.MODEL_NAME_P2, muestra_p1_p2),
        ("P3 (Grupos de Estaciones)", config.MODEL_NAME_P3, muestra_p3),
    ]:
        uri_produccion = f"models:/{nombre_modelo}@{config.PRODUCTION_ALIAS}"
        modelo_cargado = mlflow.pyfunc.load_model(uri_produccion)
        predicciones = modelo_cargado.predict(muestra)
        print(f"  -> {pregunta}")
        print(f"     URI cargada exitosamente: {uri_produccion}")
        print(f"     Predicciones de prueba (3 filas): {predicciones.tolist()}")

    print("=" * 75)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Registro en Model Registry y verificación de modelos en producción."
    )
    parser.add_argument(
        "--action",
        type=str,
        default="register_and_verify",
        choices=["register_and_verify", "summary"],
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    cliente = MlflowClient()

    experimento = cliente.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
    if experimento is None:
        raise RuntimeError(
            f"No existe el experimento '{config.MLFLOW_EXPERIMENT_NAME}' en {config.MLFLOW_TRACKING_URI}."
        )

    # Si se ejecutó desde 'mlflow run . -e register', nombramos la corrida limpiamente
    if os.environ.get("MLFLOW_RUN_ID"):
        with mlflow.start_run():
            mlflow.set_tag("mlflow.runName", f"ModelRegistry_{args.action}")
            mlflow.set_tags({"team": config.TEAM_TAG, "stage_of_work": "registry"})

    # Limpiamos cualquier corrida fallida de pruebas previas
    limpiar_corridas_fallidas(cliente, experimento.experiment_id)

    # Imprimimos el resumen de todas las corridas finalizadas con éxito
    imprimir_resumen_experimentos(cliente, experimento.experiment_id)

    if args.action == "register_and_verify":
        registrar_campeon_y_retador(
            cliente=cliente,
            experimento_id=experimento.experiment_id,
            pregunta="p1",
            nombre_registro=config.MODEL_NAME_P1,
            descripcion_modelo=(
                "Modelo de regresión para predecir el número de viajes iniciados por estación "
                "y hora en el sistema MiBici (Equipo 35). Soporta la planeación de flota."
            ),
            orden_metrica="metrics.val_mae ASC",
        )

        registrar_campeon_y_retador(
            cliente=cliente,
            experimento_id=experimento.experiment_id,
            pregunta="p2",
            nombre_registro=config.MODEL_NAME_P2,
            descripcion_modelo=(
                "Modelo de clasificación para predecir si una estación de MiBici entrará en riesgo "
                "de desbalance operativo (>= 30% de su capacidad) en la siguiente hora (Equipo 35)."
            ),
            orden_metrica="metrics.val_f1 DESC",
        )

        registrar_campeon_y_retador(
            cliente=cliente,
            experimento_id=experimento.experiment_id,
            pregunta="p3",
            nombre_registro=config.MODEL_NAME_P3,
            descripcion_modelo=(
                "Modelo de clustering de estaciones de MiBici según patrones de uso habitual, "
                "capacidad y ubicación geográfica en el AMG (Equipo 35)."
            ),
            orden_metrica="metrics.train_silhouette DESC",
        )

        verificar_carga_modelos_produccion()


if __name__ == "__main__":
    main()