import numpy as np
import pandas as pd

from mibici_demand.config import MONITORING_PARAMS as P
from mibici_demand.monitoring import alert, scenarios


def _data(n=1000):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "departures": rng.poisson(3, n),
            "lag_24": rng.poisson(3, n).astype("float32"),
            "lag_168": rng.poisson(3, n).astype("float32"),
        }
    )


def test_escenarios_alteran_solo_lo_que_simulan():
    df, s = _data(), scenarios(P)
    rng = lambda: np.random.default_rng(0)

    shifted = s["media_desplazada"](df, rng())
    assert (shifted["lag_24"] == df["lag_24"] * P["mean_shift_factor"]).all()
    assert shifted["departures"].equals(df["departures"])  # la demanda real no cambia

    missing = s["faltantes"](df, rng())
    assert abs(missing["lag_168"].isna().mean() - P["missing_share"]) < 0.05

    seasonal = s["estacional"](df, rng())
    assert seasonal["departures"].mean() < df["departures"].mean() * 0.7
    assert seasonal["lag_168"].equals(df["lag_168"])  # hace una semana aún no había vacaciones


def test_alerta_por_sesgo_aunque_el_mae_no_empeore():
    assert alert(-0.4, 64.6, 0.125, 0.0, P)[0] == "ALERTA"  # caso estacional


def test_nulos_disparan_aviso_y_revision_del_pipeline():
    assert alert(0.8, -0.1, 0.0, 0.30, P) == (
        "AVISO",
        "Revisar feature pipeline: variables faltantes",
    )
    assert alert(0.0, 4.8, 0.0, 0.0, P) == ("OK", "Ninguna")
