from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from mibici_demand.config import (
    INTERIM_DATA_DIR,
    INVALID_ID_VALUES,
    RAW_TRIPS_DIR,
)

app = typer.Typer(help="Descarga y limpieza de los viajes de MiBici.")


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


@app.command()
def main(
    input_dir: Path = RAW_TRIPS_DIR,
    output_dir: Path = INTERIM_DATA_DIR / "standardized",
) -> None:
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


if __name__ == "__main__":
    app()
