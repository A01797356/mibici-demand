from mibici_demand.download import filtrar_enlaces

BASE = "https://mibici.net/site/assets/files/"


def test_filtrar_enlaces_por_patron_conserva_catalogo():
    enlaces = {
        BASE + "datos_abiertos_2023_12.csv",
        BASE + "datos_abiertos_2024_01.csv",
        BASE + "nomenclatura_2026_09.csv",
    }
    nombres = [u.rsplit("/", 1)[1] for u in filtrar_enlaces(enlaces, "datos_abiertos_2024_*.csv")]
    assert nombres == ["datos_abiertos_2024_01.csv", "nomenclatura_2026_09.csv"]
