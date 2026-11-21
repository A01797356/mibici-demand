from types import SimpleNamespace
import pandas as pd

from mibici_demand.features import TARGET
from mibici_demand.modeling import predict


class FakeModel:
    """Sustituye al campeón del Registry: predice lag_24, sin MLflow ni red."""

    feature_names_in_ = ("lag_24",)

    def predict(self, X):
        return X["lag_24"].to_numpy()


def test_predict_genera_una_prediccion_por_fila_de_test(tmp_path, monkeypatch):
    features = pd.DataFrame(
        {
            "station_id": 1,
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            TARGET: range(10),
            "lag_24": range(10),
        }
    )
    features.to_parquet(tmp_path / "features.parquet")
    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        sklearn=SimpleNamespace(load_model=lambda uri: FakeModel()),
    )
    monkeypatch.setattr(predict, "mlflow", fake_mlflow)
  

    predict.main(tmp_path / "features.parquet", tmp_path / "pred.parquet", "models:/fake/1")
    out = pd.read_parquet(tmp_path / "pred.parquet")

    assert len(out) == 2  # último 20 % del periodo
    assert {"station_id", "timestamp", TARGET, "prediction"} <= set(out.columns)
    assert out["prediction"].tolist() == out[TARGET].tolist()
