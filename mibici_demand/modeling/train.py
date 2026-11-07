"""Entrenamiento, ajuste y registro de modelos de demanda por estación (horizonte 24 h).

Pasos:
1. Lee features.parquet y aplica el mismo split temporal de features.py.
2. Excluye las variables que no se conocen con `horizon_h` de anticipación (evita data leakage).
3. Baseline estacional: la demanda de la misma hora de la semana anterior (lag_168).
4. Cada candidato es un Pipeline (preprocesador + modelo). Se ajusta con RandomizedSearchCV y
   TimeSeriesSplit sobre una muestra de train y se reentrena con todo train.
5. Cada modelo es un run de MLflow (parámetros, métricas, resultados del tuning y modelo) y se
   registra en el Model Registry; el de menor MAE en test recibe el alias `champion`.

Comando:
    python -m mibici_demand.modeling.train --version v1
"""

from loguru import logger
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import ParameterGrid, RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
import typer

from mibici_demand.config import (
    EXPERIMENT_TRAIN,
    FEATURE_PARAMS,
    FEATURES_FILE,
    MODEL_PARAMS,
    SEARCH_SPACES,
)
from mibici_demand.features import TARGET, build_preprocessor, temporal_split
from mibici_demand.tracking import mlflow_run

app = typer.Typer(help="Entrenamiento y registro de modelos de demanda.")

BASELINE_COLUMN = "lag_168"
# Tipos que skops no confía por defecto. Son seguros porque el modelo lo genera este mismo
# pipeline; no cargar con esta lista modelos de origen desconocido.
SKOPS_TRUSTED_TYPES = [
    "numpy.dtype",
    "sklearn.model_selection._split.KFold",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
]

# Modelos candidatos. Ridge: referencia lineal e interpretable.
# HistGradientBoosting con pérdida Poisson: no lineal, escala a millones de filas y modela
# conteos (la demanda es un conteo >= 0 con ~52 % de ceros).
CANDIDATES = {
    "ridge": lambda random_state: Ridge(),
    "hgb_poisson": lambda random_state: HistGradientBoostingRegressor(
        loss="poisson", random_state=random_state
    ),
}


def leakage_columns(columns, horizon_h: int) -> list[str]:
    """Variables que usan horas posteriores a t - horizon_h y no se conocerían al predecir."""
    # ponytail: rolling_mean_24 se descarta completa porque incluye t-1; si hace falta,
    # recalcularla en features.py desplazada horizon_h horas.
    return [
        c
        for c in columns
        if (c.startswith("lag_") and int(c.removeprefix("lag_")) < horizon_h)
        or c.startswith("rolling_")
    ]


def feature_columns(columns, horizon_h: int) -> list[str]:
    """Columnas de entrada del modelo: las mismas de features.py, sin las que generan leakage."""
    excluded = {TARGET, "timestamp", "hour", "dayofweek", *leakage_columns(columns, horizon_h)}
    return [c for c in columns if c not in excluded]


def build_pipeline(model, columns: list[str]) -> Pipeline:
    """Un solo objeto para fit/predict: preprocesamiento de features.py + modelo."""
    return Pipeline([("preprocessor", build_preprocessor(columns)), ("model", model)])


def evaluate(y_true, y_pred, baseline_mae: float | None = None) -> dict[str, float]:
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }
    if baseline_mae:
        metrics["mae_improvement_vs_baseline_pct"] = 100 * (1 - metrics["mae"] / baseline_mae)
    return metrics


def tune(name: str, train: pd.DataFrame, columns: list[str], params: dict) -> RandomizedSearchCV:
    """RandomizedSearchCV con TimeSeriesSplit sobre una muestra de train ordenada en el tiempo."""
    n_rows = min(len(train), params["tune_max_rows"])
    sample = train.sample(n=n_rows, random_state=params["random_state"]).sort_values("timestamp")
    space = SEARCH_SPACES[name]
    search = RandomizedSearchCV(
        build_pipeline(CANDIDATES[name](params["random_state"]), columns),
        space,
        n_iter=min(params["n_iter"], len(ParameterGrid(space))),
        cv=TimeSeriesSplit(n_splits=params["cv_splits"]),
        scoring="neg_mean_absolute_error",
        refit=False,
        random_state=params["random_state"],
    )
    return search.fit(sample[columns], sample[TARGET])


def cv_table(search: RandomizedSearchCV) -> pd.DataFrame:
    cols = [c for c in search.cv_results_ if c.startswith("param_")]
    table = pd.DataFrame(search.cv_results_)[[*cols, "mean_test_score", "std_test_score"]]
    table["cv_mae"] = -table.pop("mean_test_score")
    return table.sort_values("cv_mae")


@app.command()
def main(
    version: str = typer.Option("v1", help="Versión de features con la que se entrena"),
    track: bool = typer.Option(True, help="Registrar runs y modelos en MLflow"),
):
    params = MODEL_PARAMS
    features = pd.read_parquet(FEATURES_FILE)
    train, test = temporal_split(features, FEATURE_PARAMS["test_fraction"])
    train = train.sort_values("timestamp", ignore_index=True)
    columns = feature_columns(features.columns, params["horizon_h"])
    logger.info(
        f"Train {train.shape} | Test {test.shape} | horizonte {params['horizon_h']} h | "
        f"excluidas por leakage: {leakage_columns(features.columns, params['horizon_h'])}"
    )

    common_params = {
        "data_version": version,
        "horizon_h": params["horizon_h"],
        "features": ",".join(columns),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
    }
    common_tags = {"stage": "training", "data_version": version}
    dataset = mlflow.data.from_pandas(
        features, source=str(FEATURES_FILE), name=f"features_{version}"
    )

    # Baseline: misma hora de la semana anterior
    baseline = evaluate(test[TARGET], test[BASELINE_COLUMN])
    tags = {**common_tags, "model": "baseline_lag168"}
    with mlflow_run(EXPERIMENT_TRAIN, f"baseline_lag168_{version}", enabled=track, tags=tags):
        if track:
            mlflow.log_params({**common_params, "model": "baseline_lag168"})
            mlflow.log_metrics(baseline)
            mlflow.log_input(dataset, context="training")
    logger.info(f"baseline_lag168: {baseline}")

    results = {}
    for name, make_model in CANDIDATES.items():
        logger.info(f"Ajustando {name}...")
        search = tune(name, train, columns, params)
        pipeline = build_pipeline(make_model(params["random_state"]), columns)
        pipeline.set_params(**search.best_params_).fit(train[columns], train[TARGET])
        predictions = pipeline.predict(test[columns])
        metrics = {
            **evaluate(test[TARGET], predictions, baseline["mae"]),
            "cv_mae": -search.best_score_,
        }
        best = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
        logger.info(f"{name}: {best} | {metrics}")

        with mlflow_run(
            EXPERIMENT_TRAIN,
            f"{name}_{version}",
            enabled=track,
            tags={**common_tags, "model": name},
        ):
            if not track:
                continue
            mlflow.log_params(
                {
                    **common_params,
                    **best,
                    "model": name,
                    "tune_rows": min(len(train), params["tune_max_rows"]),
                    "cv_splits": params["cv_splits"],
                    "n_iter": search.n_iter,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_input(dataset, context="training")
            mlflow.log_text(cv_table(search).to_csv(index=False), "tuning/cv_results.csv")
            info = mlflow.sklearn.log_model(
                pipeline,
                name="model",
                registered_model_name=params["registered_model"],
                signature=infer_signature(test[columns].head(1000), predictions[:1000]),
                input_example=test[columns].head(5),
                # MLflow serializa con skops (no ejecuta código al cargar, a diferencia de pickle)
                skops_trusted_types=SKOPS_TRUSTED_TYPES,
            )
            results[name] = (metrics["mae"], info.registered_model_version)

    if results:
        champion = min(results, key=lambda n: results[n][0])
        mae, model_version = results[champion]
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(
            params["registered_model"], params["champion_alias"], model_version
        )
        client.update_model_version(
            params["registered_model"],
            model_version,
            description=f"{champion} | features_{version} | horizonte {params['horizon_h']} h | "
            f"MAE test {mae:.4f} (baseline {baseline['mae']:.4f})",
        )
        logger.success(
            f"Campeón: {champion} -> {params['registered_model']} v{model_version} "
            f"@{params['champion_alias']} (MAE {mae:.4f} vs baseline {baseline['mae']:.4f})"
        )


if __name__ == "__main__":
    app()
