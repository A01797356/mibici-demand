"""
MLflow para versionar datasets y registrar figuras
Cada versión de un dataset se registra como un run de MLflow e incluye:
- Parámetros de las transformaciones aplicadas, umbrales o periodos
- Métricas de calidad (como filas antes y después del procesamiento)
- Info del dataset de entrada (nombre, origen, esquema y huella digital)
- Artefactos con estadísticas descriptivas, esquema y una muestra de datos (incluyendo imagenes).
"""

from contextlib import contextmanager #Para abrir y cerrar una ejecución de MLflow
import json #convertir diccionarios de Python a archivos JSON
from pathlib import Path #para trabajar con rutas de archivos y directorios
import tempfile # para crear directorios temporales para generar archivos auxiliares
from loguru import logger #funciones para registrar mensajes en la consola o en los logs
import pandas as pd #para almacenar y manipular datos (DataFrames)
from mibici_demand.config import MLFLOW_TRACKING_URI # Importa la URI del servidor de seguimiento de MLflow desde la configuración del proyecto. Puede apuntar a un servidor local o remoto.


@contextmanager
def mlflow_run(
    experiment: str,
    run_name: str,
    enabled: bool = True,
    tags: dict | None = None):
    """
    Abre y administra una ejecución (run) de MLflow.
    Parámetros:
        experiment: nombre del experimento al que pertenece el run
        run_name: nombre descriptivo para identificar la ejecución
        enabled: indica si se debe activar el registro en MLflow, si es False , no inicia una ejecución de MLflow y no registra run
        tags: diccionario opcional con etiquetas para clasificar el run
    """

    # Si el registro está desactivado, no es necesario configurar MLflow
    if not enabled:
        yield None
        return

    import mlflow

    # Configura el servidor donde MLflow almacenará los experimentos, las ejecuciones, las métricas y los artefactos.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Experimento donde se guardará la nueva ejecución, si no existe, lo crea.
    mlflow.set_experiment(experiment)

    # Inicia una ejecución y garantiza que su ciclo de vida se administre, agrega las etiquetas opcionales para facilitar las búsquedas
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        # Registra el identificador único del run y su nombre para trazabilidad de logs.
        logger.info(f"MLflow run {run.info.run_id} ({experiment}/{run_name})")
        yield run


def log_dataset_version(
    df: pd.DataFrame,
    path: Path,
    name: str,
    context: str,
    params: dict | None = None,
    metrics: dict | None = None,
    log_file: bool = False,
) -> None:
    """
    Registra info de una versión de un dataset en el run activo.
    Parámetros:
        df: df de datos procesados.
        path: ruta del archivo asociado al dataset
        name: nombre del dataset
        context: describe el uso del dataset(entrenamiento, validación o prueba)
        params: parámetros de transformaciones aplicadas
        metrics: métricas
        log_file: indica si también se debe subir el archivo completo.
    """
    import mlflow

    # Registra los parámetros de procesamiento
    if params:
        mlflow.log_params(params)

    # Registra las métricas de calidad
    if metrics:
        mlflow.log_metrics(metrics)

    # Agrega una etiqueta con el nombre del dataset
    mlflow.set_tag("dataset_name", name)

    # Guarda la ruta del archivo como una etiqueta para documentar su origen
    mlflow.set_tag("dataset_path", str(path))

    dataset = mlflow.data.from_pandas(
        df,
        source=str(path),
        name=name
    )

    # Asocia el dataset de entrada con el run activo y especifica su contexto
    mlflow.log_input(dataset, context=context)

    # Crea un directorio temporal para preparar los archivos auxiliares, al terminar se elimina
    with tempfile.TemporaryDirectory() as tmp:

        tmp = Path(tmp)

        profile = df.describe(include="all").transpose() # estadísticas descriptivas, include="all" (numéricas y categóricas)

        profile["nulls"] = df.isna().sum() # cantidad de valores nulos de cada columna

        profile.to_csv(tmp / f"{name}_profile.csv") # Guarda el perfil estadístico en formato CSV

        # Construye un diccionario con el nombre de cada columna y su tipo
        schema = {col: str(dtype)for col, dtype in df.dtypes.items()}

        # Guarda el esquema en formato JSON con sangría
        (tmp / f"{name}_schema.json").write_text(
            json.dumps(schema, indent=2)
        )

        # Guarda una muestra de hasta 1,000 filas del DataFrame.
        # Para inspeccionar los datos sin guardar necesariamente el dataset completo como artefacto
        df.head(1000).to_csv(tmp / f"{name}_sample.csv",index=False)

        # Sube todos los archivos del directorio temporal a MLflow dentro de la carpeta "dataset".
        mlflow.log_artifacts(str(tmp), artifact_path="dataset")

    # El archivo completo solo se sube cuando log_file=True.
    if log_file:
        mlflow.log_artifact(
            str(path),
            artifact_path="dataset"
        )


def log_figure(fig, filename: str) -> None:
    """
    Registra una figura de Matplotlib como artefacto de MLflow.

    Parámetros:
        fig: Figure que contiene la gráfica generada.
        filename: nombre del archivo
    """
    import mlflow

    # Comprueba que actualmente exista un run activo
    if mlflow.active_run() is not None:

        # Guarda la figura como artefacto dentro de la carpeta "figures"
        mlflow.log_figure(
            fig,
            f"figures/{filename}"
        )
