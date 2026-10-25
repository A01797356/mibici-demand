#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici_demand
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

# Parámetros del pipeline. Para cmabiarlos: make data PATTERN="datos_abiertos_2023_*.csv" VERSION=v2
PATTERN ?= datos_abiertos_2024_*.csv
VERSION ?= v1
MLFLOW_PORT ?= 5000

# En Codespaces el navegador entra a MLflow por un dominio *.app.github.dev y bloquea dominios que no sean localhost
MLFLOW_HOSTS := $(if $(CODESPACES),--allowed-hosts "*",)

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



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	python$(PYTHON_VERSION) -m venv .venv
	@echo ">>> Entorno creado. Actívalo con: source .venv/bin/activate"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make dataset
.PHONY: mlflow-ui
mlflow-ui:
	mlflow server --backend-store-uri sqlite:///mlflow.db \
		--artifacts-destination ./mlartifacts --host 127.0.0.1 --port $(MLFLOW_PORT) $(MLFLOW_HOSTS)

## Descarga MiBici CSVs a data/raw/trips/ y el catálogo de estaicones a data/raw/
.PHONY: download
download:
	$(PYTHON_INTERPRETER) -m mibici_demand.download

## Archivos de limpieza -> data/interim/trips_clean.parquet 
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici_demand.dataset clean --pattern "$(PATTERN)" --version $(VERSION)

## Genera las variables para el modelo, divide los datos en entrenamiento y prueba, prepara el preprocesador y aplica PCA. Registra la versión en MLflow.
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici_demand.features --version $(VERSION)

## Genera gráficas para EDA
## y las registra en MLflow.
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici_demand.plots

## Ejecuta el proceso en orden:preparación de datos, generación de variables y creación de gráficas.
.PHONY: pipeline
pipeline: data features plots

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
