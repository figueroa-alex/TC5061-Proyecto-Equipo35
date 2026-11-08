"""
Esquemas de validación de datos de entrada y salida con Pydantic para la API REST (Equipo 35).

Se valida de forma estricta:
  - Tipos de datos y rangos operativos reales del sistema MiBici en el AMG.
  - Rechazo automático de campos desconocidos o datos personales (extra='forbid').
  - Estructura fija y documentada para todas las respuestas JSON de la API.
"""

from pydantic import BaseModel, ConfigDict, Field

# ==============================================================================
# 1. ESQUEMA DE ENTRADA PARA P1 (DEMANDA HORARIA) Y P2 (RIESGO DE DESBALANCE)
# ==============================================================================
# Contiene exactamente las 15 variables predictoras agregadas por estación y hora
# que utilizan los modelos mibici-equipo35-p1-demanda y mibici-equipo35-p2-riesgo.
class EntradaHorariaEstacion(BaseModel):
    """Variables de entrada a nivel Estación x Hora para predecir P1 (Demanda) y P2 (Riesgo)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "hora": 8,
                "dia_semana": 1,
                "mes": 9,
                "es_fin_de_semana": 0,
                "es_hora_pico": 1,
                "dpcapacity": 19.0,
                "latitude": 20.66769,
                "longitude": -103.368252,
                "lag_1_iniciados": 6.0,
                "lag_2_iniciados": 3.0,
                "lag_24_iniciados": 5.0,
                "media_movil_3_iniciados": 4.3333,
                "lag_1_terminados": 2.0,
                "lag_1_flujo_neto": 4.0,
                "lag_1_ratio_capacidad": 0.2105,
            }
        },
    )

    hora: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hora del día en formato 24 horas (0 a 23).",
    )
    dia_semana: int = Field(
        ...,
        ge=0,
        le=6,
        description="Día de la semana (0 = Lunes, 6 = Domingo).",
    )
    mes: int = Field(
        ...,
        ge=1,
        le=12,
        description="Mes del año (1 = Enero, 12 = Diciembre).",
    )
    es_fin_de_semana: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicador binario de fin de semana (1 = Sábado o Domingo, 0 = Lunes a Viernes).",
    )
    es_hora_pico: int = Field(
        ...,
        ge=0,
        le=1,
        description="Indicador binario de hora pico (1 si la hora es 7, 8, 9, 17, 18 o 19; 0 en otro caso).",
    )
    dpcapacity: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Capacidad total de anclajes de la estación (> 0).",
    )
    latitude: float = Field(
        ...,
        ge=20.0,
        le=21.5,
        description="Latitud geográfica de la estación en el Área Metropolitana de Guadalajara.",
    )
    longitude: float = Field(
        ...,
        ge=-104.0,
        le=-102.5,
        description="Longitud geográfica de la estación en el Área Metropolitana de Guadalajara.",
    )
    lag_1_iniciados: float = Field(
        ...,
        ge=0.0,
        description="Viajes iniciados en la estación en la hora inmediata anterior (t-1).",
    )
    lag_2_iniciados: float = Field(
        ...,
        ge=0.0,
        description="Viajes iniciados en la estación dos horas antes (t-2).",
    )
    lag_24_iniciados: float = Field(
        ...,
        ge=0.0,
        description="Viajes iniciados en la estación 24 observaciones antes (t-24).",
    )
    media_movil_3_iniciados: float = Field(
        ...,
        ge=0.0,
        description="Promedio móvil de viajes iniciados en las 3 horas previas (t-1, t-2 y t-3).",
    )
    lag_1_terminados: float = Field(
        ...,
        ge=0.0,
        description="Viajes terminados (llegadas) en la estación en la hora previa (t-1).",
    )
    lag_1_flujo_neto: float = Field(
        ...,
        ge=-200.0,
        le=200.0,
        description="Flujo neto de viajes (salidas menos llegadas) en la hora previa (t-1).",
    )
    lag_1_ratio_capacidad: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        description="Proporción del desbalance absoluto previo respecto a la capacidad de la estación.",
    )


# ==============================================================================
# 2. ESQUEMA DE ENTRADA PARA P3 (CLUSTERING DE PERFIL DE ESTACIONES)
# ==============================================================================
# Contiene exactamente las 9 variables del perfil operativo y geográfico por estación
# que utiliza el modelo mibici-equipo35-p3-grupos.
class EntradaPerfilEstacionP3(BaseModel):
    """Variables de perfil histórico y geográfico de una estación para P3 (Clustering)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "promedio_salidas_diarias": 42.5,
                "duracion_mediana_min": 11.2,
                "prop_fin_de_semana": 0.21,
                "prop_pico_manana": 0.24,
                "prop_pico_tarde": 0.28,
                "ratio_salidas_llegadas": 1.04,
                "dpcapacity": 19.0,
                "latitude": 20.66769,
                "longitude": -103.368252,
            }
        },
    )

    promedio_salidas_diarias: float = Field(
        ...,
        ge=0.0,
        description="Promedio diario de viajes que inician en la estación.",
    )
    duracion_mediana_min: float = Field(
        ...,
        gt=0.0,
        le=1440.0,
        description="Duración mediana en minutos de los viajes iniciados en la estación.",
    )
    prop_fin_de_semana: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proporción de viajes realizados en sábado y domingo (0.0 a 1.0).",
    )
    prop_pico_manana: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proporción de viajes realizados en hora pico matutina 07:00-09:59 (0.0 a 1.0).",
    )
    prop_pico_tarde: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Proporción de viajes realizados en hora pico vespertina 17:00-19:59 (0.0 a 1.0).",
    )
    ratio_salidas_llegadas: float = Field(
        ...,
        gt=0.0,
        description="Relación entre total de salidas y total de llegadas en la estación (> 0).",
    )
    dpcapacity: float = Field(
        ...,
        gt=0.0,
        le=100.0,
        description="Capacidad total de anclajes de la estación (> 0).",
    )
    latitude: float = Field(
        ...,
        ge=20.0,
        le=21.5,
        description="Latitud geográfica de la estación en el Área Metropolitana de Guadalajara.",
    )
    longitude: float = Field(
        ...,
        ge=-104.0,
        le=-102.5,
        description="Longitud geográfica de la estación en el Área Metropolitana de Guadalajara.",
    )


# ==============================================================================
# 3. ESQUEMAS DE SALIDA (RESPUESTAS VALIDADAS EN FORMATO JSON)
# ==============================================================================
class RespuestaPrediccionP1(BaseModel):
    """Esquema de salida para el endpoint de regresión P1 (Demanda horaria)."""

    question: str = Field(..., description="Identificador de la pregunta de negocio (p1).")
    model_name: str = Field(..., description="Nombre del modelo en el Model Registry de MLflow.")
    model_version: str = Field(..., description="Versión del modelo utilizada para la predicción.")
    model_alias: str = Field(..., description="Alias del Model Registry utilizado (@champion).")
    loaded_from: str = Field(..., description="Fuente desde la cual se cargó el modelo.")
    viajes_iniciados_predichos: float = Field(
        ...,
        ge=0.0,
        description="Número estimado de viajes que se iniciarán en la estación en la siguiente hora.",
    )


class RespuestaPrediccionP2(BaseModel):
    """Esquema de salida para el endpoint de clasificación binaria P2 (Riesgo de desbalance)."""

    question: str = Field(..., description="Identificador de la pregunta de negocio (p2).")
    model_name: str = Field(..., description="Nombre del modelo en el Model Registry de MLflow.")
    model_version: str = Field(..., description="Versión del modelo utilizada para la predicción.")
    model_alias: str = Field(..., description="Alias del Model Registry utilizado (@champion).")
    loaded_from: str = Field(..., description="Fuente desde la cual se cargó el modelo.")
    en_riesgo: int = Field(
        ...,
        ge=0,
        le=1,
        description="Predicción binaria: 1 si la estación estará en riesgo (>= 30% capacidad), 0 si no.",
    )
    etiqueta_riesgo: str = Field(
        ...,
        description="Interpretación operativa para el equipo de rebalanceo ('EN_RIESGO' o 'SIN_RIESGO').",
    )


class RespuestaPrediccionP3(BaseModel):
    """Esquema de salida para el endpoint de clustering P3 (Grupos de estaciones)."""

    question: str = Field(..., description="Identificador de la pregunta de negocio (p3).")
    model_name: str = Field(..., description="Nombre del modelo en el Model Registry de MLflow.")
    model_version: str = Field(..., description="Versión del modelo utilizada para la predicción.")
    model_alias: str = Field(..., description="Alias del Model Registry utilizado (@champion).")
    loaded_from: str = Field(..., description="Fuente desde la cual se cargó el modelo.")
    cluster_asignado: int = Field(
        ...,
        ge=0,
        description="Identificador numérico del grupo (cluster) asignado a la estación.",
    )
    descripcion_cluster: str = Field(
        ...,
        description="Resumen del grupo asignado para planeación urbana y operativa.",
    )


class RespuestaSalud(BaseModel):
    """Esquema de salida para el endpoint operativo /health."""

    status: str = Field(..., description="Estado operativo del servicio REST ('ok').")
    team: str = Field(..., description="Identificador del equipo (Equipo35).")
    tracking_uri_configurada: str = Field(
        ..., description="Valor actual de MLFLOW_TRACKING_URI en el entorno."
    )
    modelos_cargados: list[str] = Field(
        ..., description="Lista de modelos del Model Registry listos en memoria."
    )