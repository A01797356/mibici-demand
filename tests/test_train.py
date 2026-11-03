import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from mibici_demand.modeling.train import build_pipeline, feature_columns, leakage_columns


def test_leakage_excluye_lo_que_no_se_conoce_24h_antes():
    cols = ["lag_1", "lag_24", "lag_168", "rolling_mean_24", "hour_sin"]
    assert leakage_columns(cols, 24) == ["lag_1", "rolling_mean_24"]


def test_pipeline_ajusta_y_predice_sin_leakage():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "station_id": rng.integers(1, 6, n),
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "departures": rng.poisson(2, n),
            "hour": 0,
            "dayofweek": 0,
            "month": rng.integers(1, 13, n),
            "is_weekend": 0,
            "is_holiday": 0,
            **{c: rng.normal(size=n) for c in ("hour_sin", "hour_cos", "dow_sin", "dow_cos")},
            **{f"lag_{k}": rng.poisson(2, n).astype(float) for k in (1, 24, 168)},
            "rolling_mean_24": rng.normal(size=n),
        }
    )
    columns = feature_columns(df.columns, horizon_h=24)
    pipeline = build_pipeline(Ridge(), columns).fit(df[columns], df["departures"])

    assert "lag_1" not in pipeline.feature_names_in_
    assert pipeline.predict(df[columns]).shape == (n,)
