from fastapi.testclient import TestClient
import pytest

from mibici_demand import api


class FakeModel:
    """Predice lag_24 + is_weekend: comprueba que el servicio arma las columnas del modelo."""

    feature_names_in_ = ("station_id", "lag_24", "lag_168", "is_weekend", "hour_sin")

    def predict(self, X):
        return (X["lag_24"] + X["is_weekend"]).to_numpy()


@pytest.fixture
def client():
    api.app.state.model, api.app.state.version, api.app.state.stations = FakeModel(), "2", None
    yield TestClient(api.app)  # sin "with": no corre el lifespan ni toca MLflow
    api.app.state.model = None


BODY = {
    "instances": [
        {"station_id": 2, "timestamp": "2024-12-07T08:00", "lag_24": 3, "lag_168": 4},  # sábado
        {"station_id": 3, "timestamp": "2024-12-02T08:00", "lag_24": 1, "lag_168": 0},  # lunes
    ]
}


def test_predict_devuelve_una_prediccion_por_instancia_con_version(client):
    r = client.post("/predict", json=BODY)

    assert r.status_code == 200
    assert r.json()["model_version"] == "2"
    assert [p["departures"] for p in r.json()["predictions"]] == [4.0, 1.0]


def test_predict_rechaza_lags_negativos(client):
    body = {"instances": [{**BODY["instances"][0], "lag_24": -1}]}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_sin_modelo_responde_503(client):
    api.app.state.model = None
    assert client.post("/predict", json=BODY).status_code == 503
