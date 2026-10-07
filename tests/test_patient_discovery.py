"""The cohort is whatever raw files are on disk: no hard-coded A001-A245, any ID format, optional subset."""
import pytest

import src.config as config
from src.preprocessing.load_data import discover_patients, discover_raw_patients, resolve_patients


@pytest.fixture
def raw_dir(tmp_path):
    for name in ["A001", "A010", "OD_003", "P-42"]:
        (tmp_path / f"{name}.xlsx").write_bytes(b"x")
    (tmp_path / "~$A001.xlsx").write_bytes(b"lock")        # Excel lock file
    (tmp_path / ".hidden.xlsx").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("not a cgm file")
    return tmp_path


def test_no_hard_coded_patient_range_left():
    assert not hasattr(config, "PATIENT_START") and not hasattr(config, "PATIENT_END")


def test_discover_raw_patients_any_id_format(raw_dir):
    assert discover_raw_patients(raw_dir) == ["A001", "A010", "OD_003", "P-42"]


def test_resolve_patients_defaults_to_everything(raw_dir):
    assert resolve_patients(None, raw_dir) == ["A001", "A010", "OD_003", "P-42"]


def test_resolve_patients_subset(raw_dir):
    assert resolve_patients(["OD_003", "A001", "A001"], raw_dir) == ["A001", "OD_003"]


def test_resolve_patients_rejects_unknown_id(raw_dir):
    with pytest.raises(SystemExit, match="A999"):
        resolve_patients(["A001", "A999"], raw_dir)


def test_empty_raw_folder_gives_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="No raw CGM file"):
        resolve_patients(None, tmp_path)


def test_discover_patients_in_preprocessed_folder_any_id(tmp_path):
    for pid in ["A001", "OD_002"]:
        (tmp_path / pid).mkdir()
        (tmp_path / pid / f"{pid}_win1_x_to_y_cov99pct.xlsx").write_bytes(b"x")
    (tmp_path / "empty").mkdir()
    (tmp_path / "_tmp").mkdir()
    assert discover_patients(str(tmp_path)) == ["A001", "OD_002"]
