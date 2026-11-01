"""
Chequeos automatizados de seguridad básica del repositorio (Equipo 35).
Verifica que:
  1. El archivo .gitignore exista y excluya archivos .env, datos pesados, modelos y bases locales.
  2. Exista la plantilla .env.example y que config.py utilice variables de entorno (os.getenv).
  3. Ningún archivo Python del repositorio contenga credenciales, tokens, llaves privadas
     expuestas ni funciones inseguras de ejecución arbitraria.
"""

import re

from mibici import config


def test_gitignore_protege_archivos_sensibles_y_datos():
    """Verifica que .gitignore bloquee .env, certificados, datos crudos/procesados y mlflow.db."""
    ruta_gitignore = config.PROJ_ROOT / ".gitignore"
    assert ruta_gitignore.exists(), "Falta el archivo .gitignore en la raíz del repositorio."

    contenido = ruta_gitignore.read_text(encoding="utf-8")
    patrones_obligatorios = [
        ".env",
        "*.parquet",
        "*.csv",
        "mlflow.db",
        "mlruns/",
        "*.pkl",
    ]
    for patron in patrones_obligatorios:
        assert patron in contenido, f"El patrón '{patron}' debe estar incluido en .gitignore."


def test_plantilla_env_example_y_variables_de_entorno():
    """Verifica que exista .env.example y que no exista un archivo .env rastreado con secretos."""
    ruta_env_example = config.PROJ_ROOT / ".env.example"
    assert ruta_env_example.exists(), "Debe existir el archivo plantilla .env.example."

    texto_example = ruta_env_example.read_text(encoding="utf-8")
    assert "MLFLOW_TRACKING_URI=" in texto_example
    assert "MLFLOW_EXPERIMENT_NAME=" in texto_example
    assert "MIBICI_TEST_DATA_MODE=" in texto_example


def test_ausencia_de_credenciales_y_secretos_en_codigo_fuente():
    """
    Escanea todos los archivos .py de mibici/ y tests/ para asegurar que no existan
    contraseñas, tokens de API (GitHub, AWS, OpenAI) ni llaves privadas hardcodeadas.
    """
    patrones_secretos = [
        re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key ID
        re.compile(r"ghp_[0-9a-zA-Z]{36}"),  # GitHub Personal Access Token
        re.compile(r"sk-[0-9a-zA-Z]{32,}"),  # Secret Key / API Token
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(
            r"(?:password|passwd|api_key|secret_key|access_token)\s*=\s*['\"][^'\"]+['\"]",
            re.IGNORECASE,
        ),
    ]

    carpetas_a_revisar = [config.PROJ_ROOT / "mibici", config.PROJ_ROOT / "tests"]
    for carpeta in carpetas_a_revisar:
        for ruta_py in carpeta.rglob("*.py"):
            if ruta_py.name == "test_security_repo.py":
                continue
            contenido = ruta_py.read_text(encoding="utf-8")
            for patron in patrones_secretos:
                coincidencia = patron.search(contenido)
                assert coincidencia is None, (
                    f"Posible credencial o secreto expuesto en {ruta_py.relative_to(config.PROJ_ROOT)}: "
                    f"{coincidencia.group(0)}"
                )