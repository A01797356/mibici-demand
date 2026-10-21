from mibici_demand.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, CLEANING_PARAMS


def test_project_paths():
    assert RAW_DATA_DIR.name == "raw"
    assert PROCESSED_DATA_DIR.name == "processed"
    assert RAW_DATA_DIR.exists()

def test_cleaning_params_are_consistent():
    assert CLEANING_PARAMS["min_duration_min"] < CLEANING_PARAMS["max_duration_h"] * 60
    assert CLEANING_PARAMS["min_age"] < CLEANING_PARAMS["max_age"]
