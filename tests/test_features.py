import pandas as pd

from mibici_demand.features import (
    TARGET,
    add_calendar_features,
    add_lag_features,
    aggregate_demand,
    temporal_split,
)


def _trips(rows):
    return pd.DataFrame(rows, columns=["origin_id", "start_time"]).astype(
        {"start_time": "datetime64[ns]"}
    )


def test_aggregate_demand_cuenta_salidas_y_rellena_horas_sin_viajes():
    trips = _trips(
        [
            (1, "2024-01-01 08:10"),
            (1, "2024-01-01 08:50"),
            (1, "2024-01-01 10:05"),
            (2, "2024-01-01 09:30"),
        ]
    )
    demand = aggregate_demand(trips).set_index(["station_id", "timestamp"])[TARGET]

    assert demand[(1, pd.Timestamp("2024-01-01 08:00"))] == 2
    assert demand[(1, pd.Timestamp("2024-01-01 09:00"))] == 0  # hora sin viajes = 0
    # La estación 2 entra a la malla desde su primer viaje (09:00), no antes.
    assert (2, pd.Timestamp("2024-01-01 08:00")) not in demand.index


def test_calendar_features_marca_fin_de_semana_y_festivo():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-06 23:00"])})
    out = add_calendar_features(df)

    assert out["is_holiday"].tolist() == [1, 0]  # 1-ene es festivo
    assert out["is_weekend"].tolist() == [0, 1]  # 6-ene-2024 es sábado
    assert out[["hour_sin", "hour_cos"]].abs().max().max() <= 1


def test_lags_usan_solo_el_pasado_de_la_misma_estacion():
    df = pd.DataFrame(
        {
            "station_id": [1] * 3 + [2] * 3,
            "timestamp": list(pd.date_range("2024-01-01", periods=3, freq="h")) * 2,
            TARGET: [1, 2, 3, 10, 20, 30],
        }
    )
    out = add_lag_features(df, lags=[1])

    assert len(out) == 4  # se descarta la primera hora de cada estación (sin historia)
    assert out[TARGET].sub(out["lag_1"]).isin([1, 10]).all()


def test_temporal_split_no_mezcla_pasado_y_futuro():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=10, freq="h")})
    train, test = temporal_split(df, test_fraction=0.2)

    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()
