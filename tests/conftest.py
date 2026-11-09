"""
Configuración compartida de PyTest (fixtures y puente de compatibilidad para DeepChecks).
Proporciona datasets sintéticos pequeños y deterministas (con semilla fija = 42)
para probar todas las funciones clave del proyecto MiBici sin depender de archivos externos.
"""

import sys
import types
import warnings

import numpy as np
import pandas as pd
from packaging.version import parse as parse_version
import pytest
from sklearn.metrics._scorer import _SCORERS

from mibici import config

# ==============================================================================
# 1. PUENTE DE COMPATIBILIDAD PARA DEEPCHECKS 0.19.1 EN PYTHON 3.14 / NUMPY 2.5
# ==============================================================================
# Evita bajar setuptools a versiones con vulnerabilidades (<81) al proveer parse_version
# directamente desde la librería oficial 'packaging', y registra el alias 'max_error'
# que en scikit-learn 1.9+ fue renombrado a 'neg_max_error'.
if "pkg_resources" not in sys.modules:
    modulo_pkg = types.ModuleType("pkg_resources")
    modulo_pkg.parse_version = parse_version
    sys.modules["pkg_resources"] = modulo_pkg

if "max_error" not in _SCORERS and "neg_max_error" in _SCORERS:
    _SCORERS["max_error"] = _SCORERS["neg_max_error"]

for atributo_viejo, atributo_nuevo in [
    ("Inf", "inf"),
    ("NaN", "nan"),
    ("float_", "float64"),
]:
    if not hasattr(np, atributo_viejo):
        setattr(np, atributo_viejo, getattr(np, atributo_nuevo))

if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = pd.DataFrame.map
if not hasattr(pd.Series, "iteritems"):
    pd.Series.iteritems = pd.Series.items

# En NumPy 2.x, np.quantile cambió el parámetro 'interpolation' por 'method'
_quantile_original = np.quantile


def _quantile_compatible(*args, **kwargs):
    if "interpolation" in kwargs:
        kwargs["method"] = kwargs.pop("interpolation")
    return _quantile_original(*args, **kwargs)


np.quantile = _quantile_compatible

warnings.filterwarnings(
    "ignore",
    message="The specified dataset source can be interpreted in multiple ways",
    category=UserWarning,
)


# ==============================================================================
# 2. FIXTURES DE CATÁLOGO DE ESTACIONES Y VIAJES SINTÉTICOS PARA PRUEBAS
# ==============================================================================
@pytest.fixture
def df_nomenclatura_sintetica() -> pd.DataFrame:
    """
    Crea un catálogo sintético de 6 estaciones del AMG con capacidad y coordenadas
    válidas para probar la ingeniería de características sin depender del CSV en disco.
    """
    return pd.DataFrame(
        {
            "Estacion_Id": pd.Series([10, 20, 30, 40, 50, 60], dtype="int64"),
            "name": [
                "Chapultepec Centro",
                "Arcos Vallarta",
                "Catedral Guadalajara",
                "Zapopan Basilica",
                "Tlaquepaque Parian",
                "Mexicaltzingo",
            ],
            "latitude": [20.6736, 20.6745, 20.6767, 20.7214, 20.6409, 20.6650],
            "longitude": [-103.3685, -103.3801, -103.3475, -103.3891, -103.3115, -103.3520],
            "dpcapacity": pd.Series([20, 15, 30, 25, 18, 12], dtype="int64"),
        }
    )


@pytest.fixture
def df_viajes_validos_sinteticos() -> pd.DataFrame:
    """
    Genera 610 viajes sintéticos limpios distribuidos a lo largo de 40 horas consecutivas
    en varias estaciones, garantizando suficientes rezagos (hasta lag_24) para probar
    construir_tabla_horaria_estaciones, generar_features_p1_y_p2 y generar_features_p3_clustering.
    """
    generador = np.random.default_rng(config.RANDOM_SEED)
    estaciones = [10, 20, 30, 40, 50, 60]
    # Iniciamos en viernes para que las 40 horas incluyan viernes (entre semana) y sábado (fin de semana)
    fecha_base = pd.Timestamp("2025-10-03 06:00:00")

    registros = []
    viaje_id = 1000
    for indice_hora in range(40):
        inicio_hora = fecha_base + pd.Timedelta(hours=indice_hora)
        for estacion_origen in estaciones:
            num_viajes = int(generador.integers(1, 5))
            for _ in range(num_viajes):
                viaje_id += 1
                estacion_destino = int(generador.choice(estaciones))
                duracion_min = int(generador.integers(5, 35))
                inicio_viaje = inicio_hora + pd.Timedelta(minutes=int(generador.integers(0, 20)))
                fin_viaje = inicio_viaje + pd.Timedelta(minutes=duracion_min)
                registros.append(
                    {
                        "Viaje_Id": viaje_id,
                        "Usuario_Id": int(generador.integers(100, 300)),
                        "Genero": str(generador.choice(["M", "F"])),
                        "Año_de_nacimiento": int(generador.integers(1970, 2005)),
                        "Inicio_del_viaje": inicio_viaje,
                        "Fin_del_viaje": fin_viaje,
                        "Origen_Id": estacion_origen,
                        "Destino_Id": estacion_destino,
                    }
                )

    df = pd.DataFrame(registros)
    for col in ["Viaje_Id", "Usuario_Id", "Año_de_nacimiento", "Origen_Id", "Destino_Id"]:
        df[col] = df[col].astype("Int64")
    return df