#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = TC5061-Proyecto-Equipo35
PYTHON_VERSION = 3.14.3
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) mibici/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

#################################################################################
# COMANDOS DE MLFLOW PROJECT Y SERVIDOR DE TRACKING (EQUIPO 35)                 #
#################################################################################

## Levantar servidor local de MLflow respaldado por SQLite en el puerto 5000
.PHONY: mlflow-server
mlflow-server:
	$(PYTHON_INTERPRETER) -m mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

## Ejecutar ingesta reproducible mediante MLflow Project
.PHONY: mlflow-ingest
mlflow-ingest:
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 MLFLOW_EXPERIMENT_NAME=mibici-equipo35 $(PYTHON_INTERPRETER) -m mlflow run . -e ingest --env-manager=local

## Registrar modelos campeones en Model Registry y verificar carga con @champion
.PHONY: mlflow-register
mlflow-register:
	MLFLOW_TRACKING_URI=http://127.0.0.1:5000 MLFLOW_EXPERIMENT_NAME=mibici-equipo35 $(PYTHON_INTERPRETER) -m mlflow run . -e register --env-manager=local -P action=register_and_verify