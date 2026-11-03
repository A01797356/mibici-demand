"""Inferencia con el modelo campeón del MLflow Model Registry.

Carga el Pipeline completo (preprocesador + modelo) por alias, así que no repite ninguna
transformación: las columnas de entrada salen del propio modelo (feature_names_in_).

Comando:
    python -m mibici_demand.modeling.predict      # predice el periodo de test de features.parquet
"""

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error
import typer

from mibici_demand.config import (
    FEATURE_PARAMS,
    FEATURES_FILE,
    MLFLOW_TRACKING_URI,
    MODEL_PARAMS,
    PREDICTIONS_FILE,
)
from mibici_demand.features import ID_COLUMNS, TARGET, temporal_split

app = typer.Typer(help="Predicción con el modelo campeón.")

CHAMPION_URI = f"models:/{MODEL_PARAMS['registered_model']}@{MODEL_PARAMS['champion_alias']}"


@app.command()
def main(
    input_path: Path = FEATURES_FILE,
    output_path: Path = PREDICTIONS_FILE,
    model_uri: str = typer.Option(CHAMPION_URI, help="URI del modelo en MLflow"),
):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    _, test = temporal_split(pd.read_parquet(input_path), FEATURE_PARAMS["test_fraction"])

    predictions = test[ID_COLUMNS + [TARGET]].assign(
        prediction=model.predict(test[list(model.feature_names_in_)])
    )
    predictions.to_parquet(output_path, index=False)
    mae = mean_absolute_error(predictions[TARGET], predictions["prediction"])
    logger.success(
        f"{model_uri}: {len(predictions):,} predicciones -> {output_path} | MAE {mae:.4f}"
    )


if __name__ == "__main__":
    app()
