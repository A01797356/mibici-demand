"""Integración: viajes -> features -> split -> Pipeline -> predicción -> métricas (sin MLflow)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici_demand.config import FEATURE_PARAMS, MODEL_PARAMS, RANDOM_STATE
from mibici_demand.features import TARGET, build_features, temporal_split
from mibici_demand.modeling.train import build_pipeline, evaluate, feature_columns


def _synthetic_trips(n_stations=3, days=14, seed=RANDOM_STATE) -> pd.DataFrame:
    """Viajes con patrón diario: más salidas en horas pico, distinto por estación."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2024-03-01", periods=days * 24, freq="h")
    rows = []
    for station in range(1, n_stations + 1):
        rate = 1 + station * (1 + np.sin(2 * np.pi * (hours.hour - 8) / 24))
        for hour, n in zip(hours, rng.poisson(rate)):
            minutes = rng.integers(0, 60, n)
            rows += [(station, hour + pd.Timedelta(minutes=int(m))) for m in minutes]
    return pd.DataFrame(rows, columns=["origin_id", "start_time"])


def test_pipeline_completo_de_viajes_a_metricas(tmp_path):
    # 1. Carga
    _synthetic_trips().to_parquet(tmp_path / "trips_clean.parquet")
    trips = pd.read_parquet(tmp_path / "trips_clean.parquet")

    # 2. Preprocesamiento y split temporal
    features = build_features(trips)
    train, test = temporal_split(features, FEATURE_PARAMS["test_fraction"])
    columns = feature_columns(features.columns, MODEL_PARAMS["horizon_h"])

    # 3. Entrenamiento y predicción
    model = HistGradientBoostingRegressor(loss="poisson", max_iter=50, random_state=RANDOM_STATE)
    pipeline = build_pipeline(model, columns).fit(train[columns], train[TARGET])
    prediction = pipeline.predict(test[columns])

    # 4. Métricas contra el baseline estacional (lag_168)
    baseline_mae = evaluate(test[TARGET], test["lag_168"])["mae"]
    metrics = evaluate(test[TARGET], prediction, baseline_mae=baseline_mae)

    assert train["timestamp"].max() < test["timestamp"].min()
    assert "lag_1" not in columns and "rolling_mean_24" not in columns  # sin leakage a 24 h
    assert prediction.shape == (len(test),) and (prediction >= 0).all()  # Poisson: conteos >= 0
    assert {"mae", "rmse", "r2", "mae_improvement_vs_baseline_pct"} <= metrics.keys()
    assert metrics["mae"] < baseline_mae  # el modelo aprende el patrón mejor que el baseline
