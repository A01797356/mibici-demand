"""Construcción de la variable objetivo y de las features de demanda.
Pasos:
1. Agregar los viajes limpios a salidas por estación y hora (variable objetivo).
2. Agregar variables de calendario, lags y ubicación de estación.
3. Dividir en train/test de forma temporal.
4. Ajustar el preprocesador (escalado + codificación) con train.
5. PCA de los perfiles semanales de las estaciones (reducción de dimensionalidad).
Comando:
    python -m mibici_demand.features --version v1
"""
from datetime import date
from pathlib import Path
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
import typer
from mibici_demand.config import (
    CLEAN_TRIPS_FILE,
    EXPERIMENT_DATA,
    FEATURE_PARAMS,
    FEATURES_FILE,
    PREPROCESSOR_FILE,
    PROCESSED_DATA_DIR,
    STATIONS_FILE,
    TEST_FILE,
    TRAIN_FILE,
)
from mibici_demand.dataset import load_stations
from mibici_demand.tracking import log_dataset_version, mlflow_run

app = typer.Typer(help="Features de demanda por estación.")

TARGET = "departures"
ID_COLUMNS = ["station_id", "timestamp"]

# Variable objetivo
def aggregate_demand(trips: pd.DataFrame, freq: str = "h") -> pd.DataFrame:
    """Cuenta salidas por estación y periodo, incluyendo los periodos con 0 viajes.
    Cada estación entra a la malla desde su primer viaje observado."""
    counts = (
        trips.assign(timestamp=trips["start_time"].dt.floor(freq))
        .groupby(["origin_id", "timestamp"], observed=True)
        .size()
        .rename(TARGET)
        .reset_index()
        .rename(columns={"origin_id": "station_id"})
    )
    full_range = pd.date_range(counts["timestamp"].min(), counts["timestamp"].max(), freq=freq)
    grid = pd.MultiIndex.from_product(
        [counts["station_id"].unique(), full_range], names=ID_COLUMNS
    ).to_frame(index=False)

    first_seen = counts.groupby("station_id")["timestamp"].min().rename("first_seen")
    grid = grid.merge(first_seen, on="station_id")
    grid = grid[grid["timestamp"] >= grid["first_seen"]].drop(columns="first_seen")

    demand = grid.merge(counts, on=ID_COLUMNS, how="left")
    demand[TARGET] = demand[TARGET].fillna(0).astype("int32")
    demand["station_id"] = demand["station_id"].astype("int32")
    return demand.sort_values(ID_COLUMNS, ignore_index=True)

# Features
def _nth_monday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    offset = (7 - first.weekday()) % 7  # días hasta el primer lunes
    return date(year, month, 1 + offset + 7 * (n - 1))


def mexican_holidays(years) -> set[date]:
    """Días de descanso en mexico"""
    holidays = set()
    for y in years:
        holidays |= {
            date(y, 1, 1),
            _nth_monday(y, 2, 1),  # Constitución
            _nth_monday(y, 3, 3),  # Benito Juárez
            date(y, 5, 1),
            date(y, 9, 16),
            _nth_monday(y, 11, 3),  # Revolución
            date(y, 12, 25),
        }
    return holidays

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["dayofweek"] = ts.dt.dayofweek.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    holidays = mexican_holidays(range(ts.dt.year.min(), ts.dt.year.max() + 1))
    df["is_holiday"] = ts.dt.date.isin(holidays).astype("int8")
    # Codificación cíclica: las 23 h y las 0 h quedan cerca, igual que domingo y lunes.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24).astype("float32")
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24).astype("float32")
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7).astype("float32")
    return df

def add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Demanda de la misma estación h horas antes. Elimina filas sin historia."""
    df = df.sort_values(ID_COLUMNS).copy()
    grouped = df.groupby("station_id")[TARGET]
    for lag in lags:
        df[f"lag_{lag}"] = grouped.shift(lag).astype("float32")
    # Media móvil de las 24 h previas (shift(1) para no usar la hora actual)
    df["rolling_mean_24"] = grouped.transform(
        lambda s: s.shift(1).rolling(24, min_periods=1).mean()
    ).astype("float32")
    lag_cols = [f"lag_{lag}" for lag in lags]
    return df.dropna(subset=lag_cols).reset_index(drop=True)

def add_station_info(df: pd.DataFrame, stations: pd.DataFrame | None) -> pd.DataFrame:
    if stations is None or not {"latitude", "longitude"} <= set(stations.columns):
        return df
    info = stations[["station_id", "latitude", "longitude"]].drop_duplicates("station_id")
    return df.merge(info, on="station_id", how="left")

def build_features(
    trips: pd.DataFrame, stations: pd.DataFrame | None = None, params: dict | None = None
) -> pd.DataFrame:
    params = {**FEATURE_PARAMS, **(params or {})}
    demand = aggregate_demand(trips, params["freq"])
    demand = add_calendar_features(demand)
    demand = add_lag_features(demand, params["lags"])
    return add_station_info(demand, stations)

# Split y preprocesamiento
def temporal_split(df: pd.DataFrame, test_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """El último test_fraction del periodo es test"""
    timestamps = np.sort(df["timestamp"].unique())
    cutoff = timestamps[int(len(timestamps) * (1 - test_fraction))]
    return df[df["timestamp"] < cutoff].copy(), df[df["timestamp"] >= cutoff].copy()

def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    """Escalado de numéricas, one-hot de mes y *target encoding* de estación."""
    numeric = [c for c in columns if c.startswith(("lag_", "rolling_"))]
    numeric += [c for c in ("latitude", "longitude") if c in columns]
    passthrough = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday"]

    preprocessor = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("month", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["month"]),
            ("station", TargetEncoder(target_type="continuous", random_state=42), ["station_id"]),
            ("raw", "passthrough", passthrough),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")

def transform_split(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
    """Ajusta el preprocesador solo con train y transforma ambos conjuntos."""
    feature_cols = [
        c for c in train.columns if c not in (TARGET, "timestamp", "hour", "dayofweek")
    ]
    preprocessor = build_preprocessor(feature_cols)
    x_train = preprocessor.fit_transform(train[feature_cols], train[TARGET])
    x_test = preprocessor.transform(test[feature_cols])

    def _assemble(raw: pd.DataFrame, x: pd.DataFrame) -> pd.DataFrame:
        ids = raw[ID_COLUMNS].reset_index(drop=True)
        x = x.reset_index(drop=True).add_prefix("x_")
        return pd.concat([ids, x, raw[[TARGET]].reset_index(drop=True)], axis=1)

    return _assemble(train, x_train), _assemble(test, x_test), preprocessor


# Reducción de dimensionalidad
def station_profiles_pca(demand: pd.DataFrame, n_components: int = 5):
    """PCA sobre el perfil semanal (168 horas) de cada estación.
    Cada estación se describe por su demanda media en cada hora de la semana, normalizada."""
    hour_of_week = demand["timestamp"].dt.dayofweek * 24 + demand["timestamp"].dt.hour
    profiles = demand.assign(how=hour_of_week).pivot_table(
        index="station_id", columns="how", values=TARGET, aggfunc="mean", fill_value=0
    )
    profiles = profiles.div(profiles.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    scaled = StandardScaler().fit_transform(profiles)

    n_components = min(n_components, *scaled.shape)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pd.DataFrame(
        pca.fit_transform(scaled),
        index=profiles.index,
        columns=[f"pc{i + 1}" for i in range(n_components)],
    ).reset_index()
    return scores, pca, profiles

# CLI
@app.command()
def main(
    input_path: Path = CLEAN_TRIPS_FILE,
    stations_path: Path = STATIONS_FILE,
    version: str = typer.Option("v1", help="Etiqueta de la versión del dataset"),
    track: bool = typer.Option(True, help="Registrar la versión en MLflow"),):
    """Genera features.parquet, train/test preprocesados y el preprocesador."""
    logger.info(f"Leyendo {input_path}")
    trips = pd.read_parquet(input_path, columns=["start_time", "origin_id"])
    stations = load_stations(stations_path)

    features = build_features(trips, stations)
    features.to_parquet(FEATURES_FILE, index=False)
    logger.info(f"Features: {features.shape} -> {FEATURES_FILE}")

    train, test = temporal_split(features, FEATURE_PARAMS["test_fraction"])
    train_t, test_t, preprocessor = transform_split(train, test)
    train_t.to_parquet(TRAIN_FILE, index=False)
    test_t.to_parquet(TEST_FILE, index=False)
    joblib.dump(preprocessor, PREPROCESSOR_FILE)

    scores, pca, _ = station_profiles_pca(features, FEATURE_PARAMS["pca_components"])
    pca_path = PROCESSED_DATA_DIR / "station_profiles_pca.parquet"
    scores.to_parquet(pca_path, index=False)

    params = {
        **{k: str(v) for k, v in FEATURE_PARAMS.items()},
        "source": str(input_path),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
    }
    metrics = {
        "rows_features": len(features),
        "rows_train": len(train_t),
        "rows_test": len(test_t),
        "n_stations": features["station_id"].nunique(),
        "n_features": train_t.shape[1] - 3,
        "pct_zero_demand": round(100 * (features[TARGET] == 0).mean(), 3),
        **{f"pca_var_pc{i + 1}": float(v) for i, v in enumerate(pca.explained_variance_ratio_)},
    }
    run_tags = {"stage": "features", "version": version}
    with mlflow_run(EXPERIMENT_DATA, f"features_{version}", enabled=track, tags=run_tags):
        if track:
            import mlflow

            log_dataset_version(
                features,
                FEATURES_FILE,
                f"features_{version}",
                "features",
                params=params,
                metrics=metrics,
            )
            mlflow.log_input(
                mlflow.data.from_pandas(train_t, source=str(TRAIN_FILE), name=f"train_{version}"),
                context="train",
            )
            mlflow.log_input(
                mlflow.data.from_pandas(test_t, source=str(TEST_FILE), name=f"test_{version}"),
                context="test",
            )
            mlflow.log_artifact(str(PREPROCESSOR_FILE), artifact_path="preprocessor")
            mlflow.log_artifact(str(pca_path), artifact_path="pca")
    logger.success(f"Train {train_t.shape} | Test {test_t.shape} | {metrics}")

if __name__ == "__main__":
    app()
