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
#def main(
#    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#    input_path: Path = RAW_DATA_DIR / "dataset.csv",
#    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#    # ----------------------------------------------
#):
#    # ---- REPLACE THIS WITH YOUR OWN CODE ----
#    logger.info("Processing dataset...")
#    for i in tqdm(range(10), total=10):
#        if i == 5:
#            logger.info("Something happened for iteration 5.")
#    logger.success("Processing dataset complete.")
    # -----------------------------------------

def main(
    input_dir: Path = RAW_TRIPS_DIR,
    output_dir: Path = INTERIM_DATA_DIR / "standardized",
) -> None:
    archivos = sorted(input_dir.glob("datos_abiertos_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos mensuales en {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_filas = 0

    for archivo in tqdm(archivos, desc="Estandarizando meses"):
        datos = leer_datos(archivo)
        datos = estandarizar_datos(datos)

        destino = output_dir / f"{archivo.stem}.parquet"
        datos.to_parquet(destino, index=False)

        total_filas += len(datos)
        logger.info(f"{destino.name}: {len(datos):,} filas guardadas")

    logger.success(
        f"Estandarizacion completa: {len(archivos)} archivos, "
        f"{total_filas:,} filas."
    )


if __name__ == "__main__":
    app()
