"""Configuración central del proyecto: rutas, MLflow y parámetros de limpieza.
Todos los umbrales viven aquí para que no haya "números mágicos" en el código
y para poder registrarlos como parámetros en MLflow.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
RAW_TRIPS_DIR = RAW_DATA_DIR / "trips"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Archivos canónicos del pipeline
STATIONS_FILE = RAW_DATA_DIR / "nomenclatura.csv"
CLEAN_TRIPS_FILE = INTERIM_DATA_DIR / "trips_clean.parquet"
FEATURES_FILE = PROCESSED_DATA_DIR / "features.parquet"
TRAIN_FILE = PROCESSED_DATA_DIR / "train.parquet"
TEST_FILE = PROCESSED_DATA_DIR / "test.parquet"
PREPROCESSOR_FILE = MODELS_DIR / "preprocessor.joblib"

# /data/ está en .gitignore, así que las carpetas no existen al clonar.
for _dir in (
    RAW_TRIPS_DIR,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    EXTERNAL_DATA_DIR,
    MODELS_DIR,
    FIGURES_DIR,
):
    _dir.mkdir(parents=True, exist_ok=True)

# Fuente de datos
MIBICI_OPEN_DATA_URL = "https://mibici.net/en/open-data/"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_DATA = "mibici-data-versioning"
EXPERIMENT_EDA = "mibici-eda"

# Parámetros de limpieza (se registran en MLflow en cada versión del dataset)
CLEANING_PARAMS = {
    "min_duration_min": 1.0,  # viajes de menos de 1 min son anclajes fallidos
    "max_duration_h": 24.0,  # más de 24 h: bicicleta no anclada / error
    "min_age": 16,  # edad mínima para registrarse en MiBici
    "max_age": 90,
    "iqr_k": 3.0,  # factor IQR (sobre log-duración) para outliers estadísticos
    "remove_round_trip_under_min": 2.0,  # origen == destino y < 2 min: no hubo viaje
}

# Parámetros de features
FEATURE_PARAMS = {
    "freq": "h",  # granularidad de la demanda
    "lags": [1, 24, 168],  # 1 hora, 1 día y 1 semana antes
    "test_fraction": 0.2,  # último 20 % del periodo como test (split temporal)
    "pca_components": 5,
}

INVALID_ID_VALUES = {
    "Origen_Id": [0],
    "Destino_Id": [0],
    "Viaje_Id": [0],
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
