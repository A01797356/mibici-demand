"""Simulación de data drift y pérdida de desempeño del modelo servido.

Pasos:
1. Toma el periodo de test (validación) y saca dos muestras: referencia y conjunto actual.
2. Altera el conjunto actual con cada escenario de drift (el control queda sin cambios).
3. Evalúa el modelo y detecta drift en las variables de entrada con Evidently.
4. Compara contra el control y aplica los umbrales de alerta de MONITORING_PARAMS.

Comando:
    python -m mibici_demand.monitoring        # usa models/champion (sin servidor MLflow)
"""

from functools import partial

from evidently import Report
from evidently.presets import DataDriftPreset
from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import typer

from mibici_demand.config import (
    FEATURE_PARAMS,
    FEATURES_FILE,
    FIGURES_DIR,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    MONITORING_PARAMS,
    REPORTS_DIR,
)
from mibici_demand.features import TARGET, temporal_split
from mibici_demand.modeling.train import evaluate

app = typer.Typer(help="Simulación de data drift y alertas.")

DRIFT_DIR = REPORTS_DIR / "drift"
# month se excluye: el split temporal lo hace derivar siempre (falso positivo conocido).
MONITORED = [
    "lag_24",
    "lag_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "is_holiday",
]


def mean_shift(df: pd.DataFrame, rng, factor: float) -> pd.DataFrame:
    """Error del feature pipeline: los lags llegan inflados; la demanda real no cambia."""
    return df.assign(lag_24=df["lag_24"] * factor, lag_168=df["lag_168"] * factor)


def missing_values(df: pd.DataFrame, rng, share: float) -> pd.DataFrame:
    """Falla de una fuente: una parte de lag_168 llega vacía."""
    out = df.copy()
    out.loc[rng.random(len(out)) < share, "lag_168"] = np.nan
    return out


def seasonal_drop(df: pd.DataFrame, rng, keep: float) -> pd.DataFrame:
    """Vacaciones: la demanda real y la del día previo caen; la de hace una semana aún no."""

    def thin(s):
        return rng.binomial(s.astype("int64"), keep)

    return df.assign(**{TARGET: thin(df[TARGET]), "lag_24": thin(df["lag_24"]).astype("float32")})


def scenarios(p: dict) -> dict:
    return {
        "sin_drift": lambda df, rng: df,
        "media_desplazada": partial(mean_shift, factor=p["mean_shift_factor"]),
        "faltantes": partial(missing_values, share=p["missing_share"]),
        "estacional": partial(seasonal_drop, keep=p["seasonal_keep"]),
    }


def alert(mae_change_pct: float, bias_pct: float, drift_share: float, max_null: float, p: dict):
    """Estado y acción propuesta a partir de los umbrales de MONITORING_PARAMS."""
    perf_loss = mae_change_pct >= p["mae_alert_pct"] or abs(bias_pct) >= p["bias_alert_pct"]
    nulls = max_null > p["null_alert_share"]
    if perf_loss:
        status = "ALERTA"
    elif nulls or drift_share >= p["drift_share_alert"]:
        status = "AVISO"
    else:
        return "OK", "Ninguna"
    if nulls:
        return status, "Revisar feature pipeline: variables faltantes"
    if status == "ALERTA":
        return status, "Revisar feature pipeline; si los datos son válidos, reentrenar"
    return status, "Vigilar: drift sin pérdida de desempeño"


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, name: str) -> float:
    """Reporte HTML de Evidently; devuelve la proporción de variables con drift."""
    report = Report([DataDriftPreset()]).run(
        current_data=current[MONITORED], reference_data=reference[MONITORED]
    )
    report.save_html(str(DRIFT_DIR / f"{name}.html"))
    drifted = next(
        m for m in report.dict()["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")
    )
    return float(drifted["value"]["share"])


def plot_summary(summary: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, title in [
        (axes[0], "mae", "MAE por escenario"),
        (axes[1], "bias_pct", "Sesgo relativo de la predicción (%)"),
    ]:
        bars = ax.bar(summary["escenario"], summary[col], color="#3A6EA5", width=0.6)
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
        ax.axhline(
            summary.loc[summary["escenario"] == "sin_drift", col].iloc[0],
            color="gray",
            linestyle="--",
            linewidth=1,
        )
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Impacto del data drift en el modelo mibici-demand-24h (línea punteada: control)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "drift_performance.png", dpi=150)


@app.command()
def main(
    model_uri: str = typer.Option(str(MODELS_DIR / "champion"), help="Ruta o URI del modelo"),
):
    p = MONITORING_PARAMS
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(model_uri)
    columns = list(model.feature_names_in_)

    _, test = temporal_split(pd.read_parquet(FEATURES_FILE), FEATURE_PARAMS["test_fraction"])
    sample = test.sample(n=2 * p["sample_rows"], random_state=p["random_state"])
    reference, current = sample.iloc[: p["sample_rows"]], sample.iloc[p["sample_rows"] :]

    DRIFT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, make in scenarios(p).items():
        rng = np.random.default_rng(p["random_state"])
        data = make(current, rng)
        y_pred = model.predict(data[columns])
        metrics = evaluate(data[TARGET], y_pred)
        rows.append(
            {
                "escenario": name,
                **metrics,
                "bias_pct": 100 * (y_pred.mean() - data[TARGET].mean()) / data[TARGET].mean(),
                "drift_share": drift_report(reference, data, name),
                "max_null_share": data[MONITORED].isna().mean().max(),
            }
        )

    summary = pd.DataFrame(rows)
    base_mae = summary.loc[summary["escenario"] == "sin_drift", "mae"].iloc[0]
    summary["mae_change_pct"] = 100 * (summary["mae"] / base_mae - 1)
    summary[["estado", "accion"]] = summary.apply(
        lambda r: pd.Series(
            alert(r.mae_change_pct, r.bias_pct, r.drift_share, r.max_null_share, p)
        ),
        axis=1,
    )
    summary = summary.round(4)
    summary.to_csv(DRIFT_DIR / "summary.csv", index=False)
    plot_summary(summary)
    logger.success(f"Reportes en {DRIFT_DIR}\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    app()
