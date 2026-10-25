"""Gráficas del EDA
Comando:
    python -m mibici_demand.plots
Guarda las figuras en ``reports/figures/`` y, si hay MLflow, las registra.
"""

from pathlib import Path
from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd
import typer
from mibici_demand.config import (
    CLEAN_TRIPS_FILE,
    EXPERIMENT_EDA,
    FEATURES_FILE,
    FIGURES_DIR,
    STATIONS_FILE,
)
from mibici_demand.dataset import load_stations
from mibici_demand.features import station_profiles_pca
from mibici_demand.tracking import log_figure, mlflow_run

app = typer.Typer(help="Figuras del análisis exploratorio.")

DAY_NAMES = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

def plot_monthly_trend(trips: pd.DataFrame):
    monthly = trips.set_index("start_time").resample("MS").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly.index, monthly.values, marker="o", markersize=3)
    ax.set(title="Viajes por mes", xlabel="Mes", ylabel="Viajes")
    ax.grid(alpha=0.3)
    return fig

def plot_demand_by_hour(trips: pd.DataFrame):
    """Perfil horario entre semana vs. fin de semana (picos de oficina vs. ocio)."""
    t = trips["start_time"]
    days = t.dt.normalize()
    weekend = t.dt.dayofweek >= 5
    fig, ax = plt.subplots(figsize=(10, 4))
    for label, mask in (("Entre semana", ~weekend), ("Fin de semana", weekend)):
        n_days = days[mask].nunique() or 1
        # reindex(range(24)): las horas sin viajes (madrugada) se grafican como 0
        # en lugar de unir con una línea los puntos de antes y después.
        per_hour = t[mask].dt.hour.value_counts().reindex(range(24), fill_value=0)
        (per_hour / n_days).plot(ax=ax, marker="o", label=label)
    ax.set_xticks(range(24))
    ax.set(title="Viajes promedio por hora del día", xlabel="Hora", ylabel="Viajes / día")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_heatmap_day_hour(trips: pd.DataFrame):
    t = trips["start_time"]
    # reindex(range(24)): siempre 24 columnas, aunque haya horas sin viajes. Sin esto,
    # si faltan horas (p. ej. de 2 a 4 a. m.) las columnas se recorren y el eje X
    # muestra horas equivocadas.
    table = pd.crosstab(t.dt.dayofweek, t.dt.hour).reindex(columns=range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(table.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(24), range(24))
    ax.set_yticks(range(len(table.index)), [DAY_NAMES[i] for i in table.index])
    ax.set(title="Viajes por día de la semana y hora", xlabel="Hora")
    fig.colorbar(im, ax=ax, label="Viajes")
    return fig


def plot_duration_distribution(trips: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(trips["duration_min"].clip(upper=90), bins=90)
    median = trips["duration_min"].median()
    ax.axvline(median, color="red", linestyle="--", label=f"Mediana {median:.1f} min")
    ax.set(title="Duración de los viajes (recortada a 90 min)", xlabel="Minutos", ylabel="Viajes")
    ax.legend()
    return fig

def plot_top_stations(trips: pd.DataFrame, stations: pd.DataFrame | None = None, n: int = 20):
    top = trips["origin_id"].value_counts().head(n)
    labels = top.index.astype(str)
    if stations is not None and "name" in stations.columns:
        names = stations.set_index("station_id")["name"]
        labels = [f"{i} {str(names.get(i, ''))[:25]}" for i in top.index]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], top.values[::-1])
    ax.set(title=f"Top {n} estaciones de origen", xlabel="Viajes")
    return fig

def plot_demographics(trips: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if "gender" in trips.columns:
        trips["gender"].value_counts().plot.bar(ax=axes[0], rot=0)
        axes[0].set(title="Viajes por género", xlabel="", ylabel="Viajes")
    if "age" in trips.columns:
        axes[1].hist(trips["age"].dropna(), bins=range(16, 91))
        axes[1].set(title="Edad de los usuarios (por viaje)", xlabel="Edad")
    return fig

def plot_demand_distribution(features: pd.DataFrame):
    """Distribución de la variable objetivo: muchos ceros y cola larga."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(features["departures"].clip(upper=30), bins=31)
    pct_zero = 100 * (features["departures"] == 0).mean()
    ax.set(
        title=f"Salidas por estación-hora ({pct_zero:.1f} % son cero)",
        xlabel="Salidas (recortado a 30)",
        ylabel="Frecuencia",
        yscale="log",
    )
    return fig

def plot_pca_variance(pca):
    ratio = pca.explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(6, 4))
    idx = range(1, len(ratio) + 1)
    ax.bar(idx, ratio, label="Individual")
    ax.plot(idx, ratio.cumsum(), marker="o", color="red", label="Acumulada")
    ax.set(
        title="PCA de perfiles semanales de estaciones",
        xlabel="Componente",
        ylabel="Varianza explicada",
    )
    ax.legend()
    return fig

def plot_pca_stations(scores: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(scores["pc1"], scores["pc2"], s=12, alpha=0.7)
    ax.set(title="Estaciones en el espacio PC1-PC2", xlabel="PC1", ylabel="PC2")
    ax.grid(alpha=0.3)
    return fig

def save_figure(fig, name: str, output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    log_figure(fig, path.name)
    plt.close(fig)
    return path

@app.command()
def main(
    trips_path: Path = CLEAN_TRIPS_FILE,
    features_path: Path = FEATURES_FILE,
    stations_path: Path = STATIONS_FILE,
    output_dir: Path = FIGURES_DIR,
    track: bool = typer.Option(True, help="Registrar las figuras en MLflow"),
):
    """Genera todas las figuras del EDA."""
    trips = pd.read_parquet(trips_path)
    stations = load_stations(stations_path)

    with mlflow_run(EXPERIMENT_EDA, "eda_figures", enabled=track):
        figures = {
            "01_viajes_por_mes": plot_monthly_trend(trips),
            "02_viajes_por_hora": plot_demand_by_hour(trips),
            "03_heatmap_dia_hora": plot_heatmap_day_hour(trips),
            "04_duracion": plot_duration_distribution(trips),
            "05_top_estaciones": plot_top_stations(trips, stations),
            "06_demografia": plot_demographics(trips),
        }
        if features_path.exists():
            features = pd.read_parquet(features_path)
            scores, pca, _ = station_profiles_pca(features)
            figures["07_distribucion_demanda"] = plot_demand_distribution(features)
            figures["08_pca_varianza"] = plot_pca_variance(pca)
            figures["09_pca_estaciones"] = plot_pca_stations(scores)
        else:
            logger.warning(f"No existe {features_path}; corre primero `make features`")

        for name, fig in figures.items():
            save_figure(fig, name, output_dir)
    logger.success(f"{len(figures)} figuras guardadas en {output_dir}")

if __name__ == "__main__":
    app()
