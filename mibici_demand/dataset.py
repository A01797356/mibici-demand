"""Lectura, estandarización, limpieza y versionado de viajes de MiBici.

Flujo:
1. leer_datos: lee cada CSV mensual como texto (UTF-8 o cp1252).
2. estandarizar_datos: unifica nombres y tipos (fechas en sus dos formatos).
3. TripCleaner: marca IDs inválidos, renombra a columnas canónicas y aplica las reglas de limpieza.
4. clean (CLI): guarda (data/interim/trips_clean.parquet) y registra la versión en MLflow.

Comandos:
    python -m mibici_demand.dataset standardize   # estandariza, un parquet por mes
    python -m mibici_demand.dataset clean --pattern "datos_abiertos_2024_*.csv" --version v1
"""

from pathlib import Path
import re
import unicodedata

from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import typer

from mibici_demand.config import (
    CLEAN_TRIPS_FILE,
    CLEANING_PARAMS,
    EXPERIMENT_DATA,
    INTERIM_DATA_DIR,
    INVALID_ID_VALUES,
    RAW_TRIPS_DIR,
    STATIONS_FILE,
)
from mibici_demand.tracking import log_dataset_version, mlflow_run

app = typer.Typer(help="Descarga y limpieza de los viajes de MiBici.")

# Nombres originales (ya estandarizados por ``estandarizar_datos``) -> nombres canónicos
# que usan features.py, plots.py y las pruebas.
CANONICAL_COLUMNS = {
    "Viaje_Id": "trip_id",
    "Usuario_Id": "user_id",
    "Genero": "gender",
    "A\u00f1o_de_nacimiento": "birth_year",
    "Inicio_del_viaje": "start_time",
    "Fin_del_viaje": "end_time",
    "Origen_Id": "origin_id",
    "Destino_Id": "destination_id",
}
ESSENTIAL_COLUMNS = ["start_time", "end_time", "origin_id", "destination_id"]


# Lectura y estandarización 

def leer_datos(archivo: Path) -> pd.DataFrame:
    """Lee un CSV mensual como texto, conservando los nulos y los IDs originales."""
    with archivo.open("rb") as entrada:
        encabezado = entrada.readline()
    try:
        encabezado.decode("utf-8-sig")
        codificacion = "utf-8-sig"
    except UnicodeDecodeError:
        codificacion = "cp1252"
        logger.info(f"{archivo.name}: se leera con Windows-1252.")

    logger.info(f"Leyendo {archivo.name}")
    return pd.read_csv(archivo, encoding=codificacion, dtype="string")

def estandarizar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """Unifica columnas y tipos sin eliminar filas ni corregir valores invalidos."""
    estandarizados = datos.copy()
    estandarizados.columns = estandarizados.columns.str.strip()
    estandarizados = estandarizados.rename(
        columns={"A}\u00e4e_nacimiento": "A\u00f1o_de_nacimiento"}
    )
    if estandarizados.columns.duplicated().any():
        raise ValueError("La estandarizacion produce nombres de columnas duplicados.")

    columnas_enteras = (
        "Viaje_Id",
        "Usuario_Id",
        "Origen_Id",
        "Destino_Id",
        "A\u00f1o_de_nacimiento",
    )
    columnas_fecha = ("Inicio_del_viaje", "Fin_del_viaje")
    requeridas = {*columnas_enteras, *columnas_fecha, "Genero"}
    faltantes = requeridas - set(estandarizados.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de viajes: {sorted(faltantes)}")

    for columna in columnas_enteras:
        try:
            numericos = pd.to_numeric(estandarizados[columna], errors="raise")
            if (
                numericos.isin([float("inf"), float("-inf")]).any()
                or (numericos.notna() & numericos.mod(1).ne(0)).any()
            ):
                raise ValueError("La conversion a entero perderia informacion.")
            estandarizados[columna] = numericos.astype("Int64")
        except (ValueError, TypeError) as error:
            raise ValueError(f"{columna}: contiene valores que no son enteros validos.") from error

    for columna in columnas_fecha:
        try:
            texto = estandarizados[columna].astype("string").str.strip()
            formato_dia_primero = texto.str.contains("/", regex=False, na=False)
            fechas = pd.to_datetime(
                texto.mask(formato_dia_primero), format="%Y-%m-%d %H:%M:%S", errors="raise"
            )
            fechas.loc[formato_dia_primero] = pd.to_datetime(
                texto.loc[formato_dia_primero], format="%d/%m/%Y %H:%M", errors="raise"
            )
            estandarizados[columna] = fechas.astype("datetime64[ns]")
        except ValueError as error:
            raise ValueError(f"{columna}: contiene fechas no interpretables.") from error

    estandarizados["Genero"] = estandarizados["Genero"].astype("string").str.strip().str.upper()
    return estandarizados


def buscar_nulos(input_dir: Path = RAW_TRIPS_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve dos tablas globales: nulos originales y nulos incluyendo IDs invalidos."""
    archivos = sorted(input_dir.glob("datos_abiertos_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos mensuales en {input_dir}.")

    conteos = []
    for archivo in tqdm(archivos, desc="Nulos por mes"):
        datos = leer_datos(archivo)
        faltantes = set(INVALID_ID_VALUES) - set(datos.columns)
        if faltantes:
            raise ValueError(f"{archivo.name}: faltan columnas de IDs: {sorted(faltantes)}")

        conteo = datos.isna().sum().rename_axis("columna").to_frame("nulos")
        conteo["ids_invalidos"] = 0
        for columna, valores_invalidos in INVALID_ID_VALUES.items():
            ids_numericos = pd.to_numeric(datos[columna], errors="raise")
            mascara = datos[columna].notna() & ids_numericos.isin(valores_invalidos)
            conteo.loc[columna, "ids_invalidos"] = int(mascara.sum())
        conteo["filas"] = len(datos)
        conteos.append(conteo)

    totales = pd.concat(conteos).groupby(level="columna").sum()
    nulos_originales = totales[["nulos", "filas"]].copy()
    nulos_originales["nulos_pct"] = (
        nulos_originales["nulos"] / nulos_originales["filas"].replace(0, float("nan")) * 100
    )
    nulos_con_invalidos = totales.rename(columns={"nulos": "nulos_originales"}).copy()
    nulos_con_invalidos["nulos"] = (
        nulos_con_invalidos["nulos_originales"] + nulos_con_invalidos["ids_invalidos"]
    )
    nulos_con_invalidos = nulos_con_invalidos[
        ["nulos_originales", "ids_invalidos", "nulos", "filas"]
    ]
    nulos_con_invalidos["nulos_pct"] = (
        nulos_con_invalidos["nulos"] / nulos_con_invalidos["filas"].replace(0, float("nan")) * 100
    )
    return (
        nulos_originales.sort_values("nulos", ascending=False),
        nulos_con_invalidos.sort_values("nulos", ascending=False),
    )

# CATÁLOGO DE ESTACIONES: Lista oficial de estaciones (ID, nombre, coordenadas). Para descartar viajes con estaciones que no existen

# Posibles nombres de columna del catálogo, se comparan en minúsculas, sin acentos ni símbolos.
STATION_ALIASES = {
    "station_id": ["id", "idestacion", "stationid"],
    "name": ["name", "nombre"],
    "latitude": ["latitude", "latitud", "lat"],
    "longitude": ["longitude", "longitud", "lon", "lng"],
    "status": ["status", "estatus", "estado"],
}

def _normalize(name: str) -> str:
    """Simplifica un nombre de columna. Por ejmplo: "Latitud " a "latitud", "ID_Estación" a "idestacion"."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", text.lower())

def find_stations_file(path: Path = STATIONS_FILE) -> Path | None:
    """Devuelve el catálogo de estaciones.
    Usndo data/raw/nomenclatura.csv si existe, si no, el archivo más reciente que contenga "nomenclatura".
    """
    if path.exists():
        return path
    candidates = sorted(p for p in path.parent.glob("*.csv") if "nomenclatura" in p.name.lower())
    return candidates[-1] if candidates else None

def load_stations(path: Path = STATIONS_FILE) -> pd.DataFrame | None:
    """Busca y lee el catálogo, estandariza las columnas, valida los identificadores, 
    convierte los datos numéricos y devuelve la tabla limpia."""
    # Busca el archivo del catálogo.
    found = find_stations_file(path)
    # Si no existe, registra una advertencia y continúa sin el catálogo.
    if found is None:
        logger.warning(f"No se encontró un catálogo de estaciones en {path.parent}")
        return None
    # Lee los datos del archivo.
    raw = leer_datos(found)
    # Relaciona los nombres de columnas con catálogo.
    rename = {}
    for col in raw.columns:
        key = _normalize(col)
        for canonical, options in STATION_ALIASES.items():
            if key in options and canonical not in rename.values():
                rename[col] = canonical
                break
    if "station_id" not in rename.values():
        raise ValueError(f"{found.name}: no se encontró la columna de ID de estación")
    # Renombra las columnas y conserva solo las reconocidas
    stations = raw.rename(columns=rename)[list(rename.values())]
    # Convierte identificadores y coordenadas a números, los que no pueden convertir se marcan NaN
    for col in ("station_id", "latitude", "longitude"):
        if col in stations.columns:
            stations[col] = pd.to_numeric(stations[col], errors="coerce")
    # Elimina las filas sin identificador y convierte los ID a enteros
    stations = stations.dropna(subset=["station_id"]).astype({"station_id": "int32"})
    # Registro
    logger.info(f"Catálogo de estaciones: {found.name} ({len(stations)} estaciones)")
    # Devuelve tabla limpia
    return stations

# LIMPIEZA

# TripCleaner recibe un mes crudo y devuelve los viajes válidos.
class TripCleaner:
    """Aplica las reglas de limpieza a un DataFrame de viajes crudo y devuelve los viajes válidos.
    Cada paso es un método independiente y anota en self.report_ cuántas filas eliminó para documentar y versionar la limpieza.
    Uso:
        cleaner = TripCleaner(station_ids={1, 2, 3})
        limpio = cleaner.run(leer_datos(archivo))
        cleaner.report_  # {"rows_raw": ..., "duplicates": ..., "rows_clean": ...}
    """
    
    def __init__(self, params: dict | None = None, station_ids: set[int] | None = None):
        # guarda las reglas de limpieza predeterminadas y permite personalizarlas.
        self.params = {**CLEANING_PARAMS, **(params or {})}
        # Lista de IDs de estaciones válidas
        self.station_ids = station_ids
        # crea un diccionario vacío para registrar cuántas filas se eliminan durante la limpieza.
        self.report_: dict[str, int] = {}

    def _add(self, key: str, value: int) -> None:
        #Suma una cantidad al contador correspondiente del reporte
        self.report_[key] = self.report_.get(key, 0) + int(value)

    def _track(self, step: str, before: int, df: pd.DataFrame) -> pd.DataFrame:
        #Registra cuántas filas se eliminaron en un paso de limpieza
        self._add(step, before - len(df))
        return df

    # Estandarizar los datos.
    def standardize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Unifica los nombres de columnas y tipos de datos."""
        return estandarizar_datos(raw)

    # Marcar identificadores inválidos.
    def mark_invalid_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte los IDs inválidos en valores vacíos"""
        df = df.copy()
        for column, invalid in INVALID_ID_VALUES.items():
            if column in df.columns:
                mask = df[column].isin(invalid)
                self._add("invalid_ids_set_na", mask.sum())
                df.loc[mask, column] = pd.NA
        return df

    # Unifica nombres de las columnas
    def to_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cambia nombres de columnas a nombres estándar"""
        # Comprueba que estén todas las columnas necesarias para el renombrado.
        missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas para renombrar: {missing}")
        # Renombra columnas y conserva solo las que reconoce el proyecto.
        return df.rename(columns=CANONICAL_COLUMNS)[list(CANONICAL_COLUMNS.values())]

    # Eliminar viajes con info esencial faltante
    def drop_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina viajes sin fecha de inicio, fecha de fin, origen o destino."""
        before = len(df)
        df = df.dropna(subset=ESSENTIAL_COLUMNS)
        return self._track("missing_essentials", before, df)

    # Eliminar viajes duplicados.
    def drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas idénticas y viajes con el mismo ID."""
        before = len(df)
        df = df.drop_duplicates()
        # Los viajes sin ID no se consideran duplicados por esta regla.
        repeated_id = df["trip_id"].notna() & df["trip_id"].duplicated()
        return self._track("duplicates", before, df[~repeated_id])

    # Calcular la duración de cada viaje.
    def add_duration(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula la duración en minutos."""
        df = df.copy()
        df["duration_min"] = ((df["end_time"] - df["start_time"]).dt.total_seconds() / 60)
        return df

    # Eliminar duraciones fuera de los límites permitidos.
    def drop_invalid_durations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Conserva los viajes cuya duración está dentro del rango establecido."""
        before = len(df)
        p = self.params
        # La duración debe cumplir el mínimo y el máximo configurados.
        # Se eliminan duraciones negativas.
        valid = (
            (df["duration_min"] >= p["min_duration_min"])
            & (df["duration_min"] <= p["max_duration_h"] * 60)
        )
        return self._track("invalid_duration", before, df[valid])

    # Eliminar viajes cortos que comienzan y terminan en el mismo lugar.
    def drop_false_round_trips(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina viajes con mismo origen y destino en poco tiempo."""
        before = len(df)
        false_trip = (
            (df["origin_id"] == df["destination_id"])
            & (df["duration_min"] < self.params["remove_round_trip_under_min"]))
        return self._track("false_round_trip", before, df[~false_trip])

    # Eliminar duraciones outliers
    def remove_duration_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta duraciones atípicas con el rango intercuartílico (IQR)."""
        before = len(df)
        if df.empty:
            return df
        # Aplica logaritmo para reducir la influencia de las duraciones muy grandes
        log_d = np.log1p(df["duration_min"])
        q1, q3 = log_d.quantile([0.25, 0.75])
        # Define cuánto puede alejarse un valor del rango habitual.
        k = self.params["iqr_k"]
        mask = log_d.between(
            q1 - k * (q3 - q1),
            q3 + k * (q3 - q1)
        )
        return self._track("duration_outlier", before, df[mask])

    # Limpiar los datos demográficos.
    def clean_demographics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estandariza el género y marca las edades no válidas como desconocidas."""
        df = df.copy()
        g = df["gender"].astype("string").str.strip().str.upper().str[0]
        df["gender"] = (
            g.where(g.isin(["M", "F"]), "ND")
            .fillna("ND")
            .astype("category")
        )

        # Calcula la edad aproximada 
        age = df["start_time"].dt.year - df["birth_year"]
        # Comprueba que la edad esté dentro de los límites configurados.
        valid = age.between(
            self.params["min_age"],
            self.params["max_age"]
        ).fillna(False)
        # Cuenta las edades inválidas sin eliminar los viajes.
        self._add(
            "invalid_age_set_nan",
            (~valid & df["birth_year"].notna()).sum()
        )
        df["birth_year"] = df["birth_year"].where(valid)
        df["age"] = age.where(valid).astype("float32")
        return df

    # Comprobar que las estaciones existan en el catálogo.
    def filter_unknown_stations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina viajes cuyo origen o destino no aparece en la lista de IDs válidos"""
        if not self.station_ids:
            return df
        before = len(df)
        known = (
            df["origin_id"].isin(self.station_ids)
            & df["destination_id"].isin(self.station_ids)
        )
        return self._track("unknown_station", before, df[known])

    # Reducir el espacio que ocupan los datos en memoria.
    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Asigna tipos de datos más compactos a las columnas."""
        return df.astype(
            {
                "trip_id": "Int64",
                "user_id": "Int64",
                "birth_year": "Int16",
                "origin_id": "int32",
                "destination_id": "int32",
                "duration_min": "float32",
            }
        )

    # Ejecutar todos los pasos de limpieza.
    def run(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Aplica las reglas de limpieza en orden y devuelve los viajes válidos."""
        
        # Registra cuántas filas tenía el archivo antes de limpiarlo.
        self._add("rows_raw", len(raw))
        df = raw

        # Ejecuta cada regla en el orden establecido.
        # El orden importa: primero se calcula la duración y después se valida.
        for step in (
            self.standardize,
            self.mark_invalid_ids,
            self.to_canonical,
            self.drop_missing,
            self.drop_duplicates,
            self.add_duration,
            self.drop_invalid_durations,
            self.drop_false_round_trips,
            self.remove_duration_outliers,
            self.clean_demographics,
            self.filter_unknown_stations,
            self.cast_types,
        ):
            df = step(df)

        # Registra cuántas filas quedaron después de la limpieza.
        self._add("rows_clean", len(df))

        # Reinicia el índice para que las filas queden numeradas consecutivamente.
        return df.reset_index(drop=True)


def clean_files(files: list[Path], cleaner: TripCleaner) -> pd.DataFrame:
    """Limpia varios archivos por separado y después reúne todos los resultados."""
    parts = []
    # Procesa cada archivo por separado
    for path in tqdm(files, desc="Limpiando"):
        df = cleaner.run(leer_datos(path))
        # Guarda el nombre del archivo de origen para rastrear cada viaje.
        df["source_file"] = path.name
        parts.append(df)
    # Combina los viajes limpios de todos los archivos en una sola tabla.
    trips = pd.concat(parts, ignore_index=True)
    trips["source_file"] = trips["source_file"].astype("category")
    return trips.sort_values("start_time", ignore_index=True)

@app.command()
def standardize(
    input_dir: Path = RAW_TRIPS_DIR,
    output_dir: Path = INTERIM_DATA_DIR / "standardized",
) -> None:
    """Solo estandariza (sin limpiar): un parquet por mes en data/interim/standardized/."""
    archivos = sorted(input_dir.glob("datos_abiertos_*.csv"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos mensuales en {input_dir}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_filas = 0

    for archivo in tqdm(archivos, desc="Estandarizando meses"):
        datos = leer_datos(archivo)
        datos = estandarizar_datos(datos)

        destino = output_dir / f"{archivo.stem}.parquet"
        datos.to_parquet(destino, index=False)

        total_filas += len(datos)
        logger.info(f"{destino.name}: {len(datos):,} filas guardadas")

    logger.success(f"Estandarizacion completa: {len(archivos)} archivos, {total_filas:,} filas.")

@app.command()
def clean(
    pattern: str = "datos_abiertos_*.csv",
    version: str = "v1",
    track: bool = True,
) -> None:
    """Limpia los CSV del periodo elegido, guarda trips_clean.parquet y lo versiona en MLflow.
    Opciones:
        --pattern   archivos a limpiar, p. ej. "datos_abiertos_2024_*.csv"
        --version   etiqueta de la versión en MLflow
        --no-track  limpia sin registrar en MLflow para pruebas rápidas
    """
    # Archivos del periodo elegido.
    files = sorted(RAW_TRIPS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No hay archivos {pattern} en {RAW_TRIPS_DIR}")

    # Limpieza mes a mes. Si hay catálogo descarta estaciones inexistentes.
    stations = load_stations()
    station_ids = set(stations["station_id"]) if stations is not None else None
    cleaner = TripCleaner(station_ids=station_ids)
    trips = clean_files(files, cleaner)
    trips.to_parquet(CLEAN_TRIPS_FILE, index=False)
    logger.info(f"Reporte de limpieza: {cleaner.report_}")

    # Registro de la versión en MLflow: umbrales y periodo como parámetros, filas eliminadas por regla y dataset como input.
    if track:
        name = f"trips_clean_{version}"
        with mlflow_run(EXPERIMENT_DATA, name, tags={"version": version}):
            log_dataset_version(
                trips,
                CLEAN_TRIPS_FILE,
                name,
                "cleaning",
                params={**cleaner.params, "pattern": pattern, "n_files": len(files)},
                metrics=cleaner.report_,)
    logger.success(f"{len(trips):,} viajes limpios guardados en {CLEAN_TRIPS_FILE}")

# Para ejecutar el archivo con "python -m mibici_demand.dataset ...".
if __name__ == "__main__":
    app()
