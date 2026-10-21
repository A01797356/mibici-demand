from mibici_demand.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def test_project_paths():
    assert RAW_DATA_DIR.name == "raw"
    assert PROCESSED_DATA_DIR.name == "processed"
