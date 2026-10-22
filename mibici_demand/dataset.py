from pathlib import Path
import re
import unicodedata

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mibici_demand.config import (
    CLEAN_TRIPS_FILE,
    CLEANING_PARAMS,
    EXPERIMENT_DATA,
    RAW_TRIPS_DIR,
    STATIONS_FILE,
)
from mibici_demand.tracking import log_dataset_version, mlflow_run

app = typer.Typer(help="Descarga y limpieza de los viajes de MiBici.")

@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Processing dataset...")
    for i in tqdm(range(10), total=10):
        if i == 5:
            logger.info("Something happened for iteration 5.")
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
