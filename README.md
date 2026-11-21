# mibici-demand

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Proyecto MLOps · Equipo 03 · TC5044.10 Operaciones de aprendizaje automático (Tec de Monterrey, MNA).

Predicción de demanda de bicicletas por estación con los datos abiertos de
[MiBici](https://mibici.net/en/open-data/) (Área Metropolitana de Guadalajara).

## Integrantes

| Integrante | GitHub |
|---|---|
| Jaime Espinosa | [@A01797356](https://github.com/A01797356) |
| Samantha Rizo Franco | [@SamanthaRizo](https://github.com/SamanthaRizo) |
| Ytalo Zerpa | [@ZerpaYtalo](https://github.com/ZerpaYtalo) |
| Daniel Rico | [@DanielRCH10](https://github.com/DanielRCH10) |

## Setup

Requisitos: Python 3.12 (WSL/Linux) y un servidor MLflow accesible.

```bash
git clone https://github.com/A01797356/mibici-demand.git
cd mibici-demand
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "MLFLOW_TRACKING_URI=http://127.0.0.1:5000" > .env   # .env no se versiona
pytest
```

Los datos (`data/`) no se versionan en Git; sus versiones se registran en MLflow.

## Datos y limpieza

```bash
# Terminal 2: servidor MLflow local
mlflow server --backend-store-uri sqlite:///mlflow.db \
  --artifacts-destination ./mlartifacts --host 127.0.0.1 --port 5000

# Terminal 1
python -m mibici_demand.download --pattern "datos_abiertos_2024_*.csv" # CSV -> data/raw/trips/ y nomenclatura -> data/raw/ (sin --pattern: todo el histórico)
python -m mibici_demand.dataset standardize # opcional: estandariza un parquet por mes
python -m mibici_demand.dataset clean --pattern "datos_abiertos_2024_*.csv" --version v1
```

`clean` genera `data/interim/trips_clean.parquet` y registra la versión `trips_clean_v1` en el experimento `mibici-data-versioning` de MLflow (http://127.0.0.1:5000).

Para probar sin MLflow agregar `--no-track`.

## Reproducir el proyecto

```bash
# Terminal 2 (con el entorno activo)
make mlflow-ui

# Terminal 1
make download
cp references/nomenclatura_2026_09.csv data/raw/nomenclatura.csv   # fija el catálogo de 484 estaciones
make data features plots train predict
```

Resultado esperado: 625,551 observaciones de test y MAE ≈ 0.802.

MiBici solo publica su catálogo más reciente. Sin el paso `cp`, se usa ese catálogo y los
resultados cambian ligeramente (ver `docs/data_versioning.md`).

## Resultados

Demanda horaria por estación con 24 h de anticipación. Test: 21-oct-2024 a 31-dic-2024.

| Modelo | MAE | RMSE | R² | Mejora vs baseline |
|---|---|---|---|---|
| Baseline `lag_168` | 0.974 | 1.773 | 0.467 | — |
| Ridge | 0.922 | 1.480 | 0.629 | 5.4 % |
| **HGB Poisson (`@champion`)** | **0.802** | **1.344** | **0.694** | **17.7 %** |

Documentación: [modelos y pipeline](docs/modeling.md) ·
[experimentos y Registry](docs/experiment_tracking.md) ·
[versiones de datos](docs/data_versioning.md) · [roles](docs/roles_phase2.md) ·
[ML Canvas](references/ml_canvas.md)

## Project Organization

```
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Documentación del proyecto: modelos, experimentos, datos y roles
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_demand and configuration for tools like black
│
├── references         <- ML Canvas y catálogo de estaciones de la corrida oficial
│
├── reports            <- Figuras, exportación de runs y artefactos de la Fase 1
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── tests              <- Pruebas automatizadas (pytest)
│
└── mibici_demand   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_demand a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Lectura, estandarización y limpieza de los datos
    │
    ├── download.py             <- Descarga los CSV y el catálogo de MiBici
    │
    ├── tracking.py             <- Funciones de apoyo para registrar en MLflow
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
 
```
