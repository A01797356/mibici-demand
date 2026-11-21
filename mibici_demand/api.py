"""Servicio de predicción de demanda (FastAPI) con el modelo del MLflow Model Registry.

Variables de entorno:
    MODEL_URI      models:/mibici-demand-24h@champion (default) o una ruta local al modelo
    MODEL_VERSION  versión a reportar cuando MODEL_URI es una ruta local (en Docker)

Comando:
    uvicorn mibici_demand.api:app --port 8000      # Swagger en http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from datetime import datetime
import os

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from mibici_demand.config import MLFLOW_TRACKING_URI, MODEL_PARAMS, PROJ_ROOT
from mibici_demand.dataset import load_stations
from mibici_demand.features import add_calendar_features, add_station_info

CHAMPION_URI = f"models:/{MODEL_PARAMS['registered_model']}@{MODEL_PARAMS['champion_alias']}"
MODEL_URI = os.getenv("MODEL_URI", CHAMPION_URI)
STATIONS_REF = PROJ_ROOT / "references" / "nomenclatura_2026_09.csv"


class Instance(BaseModel):
    station_id: int = Field(..., examples=[2])
    timestamp: datetime = Field(..., description="Hora a predecir", examples=["2024-12-02T08:00"])
    lag_24: float = Field(..., ge=0, description="Salidas de la estación 24 h antes", examples=[3])
    lag_168: float = Field(..., ge=0, description="Salidas 168 h (1 semana) antes", examples=[4])


class PredictRequest(BaseModel):
    instances: list[Instance] = Field(..., min_length=1)


class Prediction(BaseModel):
    station_id: int
    timestamp: datetime
    departures: float


class PredictResponse(BaseModel):
    model_uri: str
    model_version: str
    predictions: list[Prediction]


def resolve_version(uri: str) -> str:
    """Número de versión del Registry detrás de un alias; si no aplica, MODEL_VERSION."""
    if uri.startswith("models:/") and "@" in uri:
        name, alias = uri.removeprefix("models:/").split("@")
        return str(mlflow.MlflowClient().get_model_version_by_alias(name, alias).version)
    return os.getenv("MODEL_VERSION", "unknown")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model = mlflow.sklearn.load_model(MODEL_URI)
    app.state.version = resolve_version(MODEL_URI)
    app.state.stations = load_stations(STATIONS_REF)
    yield


app = FastAPI(
    title="MiBici demand API",
    description="Salidas por estación con 24 h de anticipación (HGB Poisson, Equipo 03).",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI, "model_version": app.state.version}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")

    df = pd.DataFrame([i.model_dump() for i in request.instances])
    df = add_station_info(add_calendar_features(df), app.state.stations)
    try:
        y = model.predict(df[list(model.feature_names_in_)])
    except Exception as exc:  # columnas o tipos inesperados: error del cliente, no caída
        raise HTTPException(status_code=422, detail=f"No se pudo predecir: {exc}") from exc

    return PredictResponse(
        model_uri=MODEL_URI,
        model_version=app.state.version,
        predictions=[
            Prediction(station_id=s, timestamp=t, departures=round(float(p), 3))
            for s, t, p in zip(df["station_id"], df["timestamp"], y)
        ],
    )
