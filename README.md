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
| Samantha Rizo Franco | [@A01796848]([https://github.com/A01797356](https://github.com/SamanthaRizo)|

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
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
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_demand   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_demand a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
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

--------

