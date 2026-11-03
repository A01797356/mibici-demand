from mibici_demand.dataset import leer_datos


def test_leer_datos_detecta_cp1252_con_encabezado_ascii(tmp_path):
    archivo = tmp_path / "nomenclatura.csv"
    archivo.write_bytes("id,name\r\n3,Av. Niños héroes\r\n".encode("cp1252"))
    assert leer_datos(archivo).loc[0, "name"] == "Av. Niños héroes"
