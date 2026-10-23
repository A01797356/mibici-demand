import csv
from html.parser import HTMLParser
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from urllib.parse import urljoin, urlparse

from loguru import logger

from mibici_demand.config import MIBICI_OPEN_DATA_URL, RAW_DATA_DIR, RAW_TRIPS_DIR


class EnlacesCSV(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.enlaces: set[str] = set()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for atributo, valor in attrs:
            if atributo == "href" and valor:
                url = urljoin(MIBICI_OPEN_DATA_URL, valor)
                if urlparse(url).path.lower().endswith(".csv"):
                    self.enlaces.add(url)


def descargar_csvs() -> None:
    """Descarga los CSV publicados por MiBici sin sobrescribir archivos existentes."""
    opciones_curl = [
        "curl",
        "--fail",
        "--silent",
        "--show-error",
        "--location",
        "--retry",
        "3",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
    ]
    pagina = subprocess.run(
        [*opciones_curl, MIBICI_OPEN_DATA_URL],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    parser = EnlacesCSV()
    parser.feed(pagina.stdout)
    if not parser.enlaces:
        raise ValueError("No se encontraron enlaces CSV en la pagina.")

    for numero, url in enumerate(sorted(parser.enlaces), start=1):
        nombre = Path(urlparse(url).path).name
        carpeta = RAW_TRIPS_DIR if nombre.startswith("datos_abiertos_") else RAW_DATA_DIR
        carpeta.mkdir(parents=True, exist_ok=True)
        destino = carpeta / nombre
        if destino.exists():
            logger.warning(f"Ya existe; se omite sin verificar: {nombre}")
            continue

        with NamedTemporaryFile(
            dir=carpeta, prefix=f".{nombre}.", suffix=".part", delete=False
        ) as tmp:
            temporal = Path(tmp.name)
        try:
            subprocess.run(
                [*opciones_curl, "--output", str(temporal), url],
                check=True,
            )
            if temporal.stat().st_size == 0:
                raise ValueError(f"Archivo descargado vacio: {nombre}")

            with temporal.open("rb") as entrada:
                primera_linea = entrada.readline()
            try:
                texto = primera_linea.decode("utf-8-sig")
            except UnicodeDecodeError:
                texto = primera_linea.decode("cp1252")
            encabezado = next(csv.reader([texto]))
            if len(encabezado) < 2 or any("<html" in campo.lower() for campo in encabezado):
                raise ValueError(f"La descarga no parece un CSV: {nombre}")

            if destino.exists():
                raise FileExistsError(f"El archivo aparecio durante la descarga: {destino}")
            temporal.rename(destino)
            logger.info(f"[{numero}/{len(parser.enlaces)}] Descargado: {nombre}")
        finally:
            temporal.unlink(missing_ok=True)


if __name__ == "__main__":
    descargar_csvs()
