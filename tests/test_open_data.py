"""Open-dataset anonymisation, its conversion back to the raw layout, and the synthetic sample data."""
import numpy as np
import pandas as pd
import pytest

from scripts.preprocessing import generate_open_data as od
from scripts.preprocessing import open_data_to_raw as to_raw
from scripts.sample_data import generate_sample_data as sample


@pytest.fixture
def window():
    t = pd.date_range("2024-05-06 08:13", periods=500, freq="5min")        # a "real-looking" start date
    return pd.DataFrame({"time": t, "glycemie": np.linspace(80, 220, 500)})


def test_anonymised_ids_are_sequential_and_padded():
    assert od._make_anon_ids(3) == ["OD_001", "OD_002", "OD_003"]


def test_anonymisation_removes_absolute_time(window):
    out = od.anonymise_patient(window, "OD_001", noise_std=2.0, rng=np.random.default_rng(0), phase_shift=False)
    assert list(out.columns) == ["patient_id", "time_h", "glucose_mgdl"]
    assert out["time_h"].iloc[0] == 0.0
    assert out["time_h"].iloc[1] == pytest.approx(5 / 60, abs=1e-3)
    assert set(out["patient_id"]) == {"OD_001"}


def test_noise_is_small_reproducible_and_clipped(window):
    a = od.anonymise_patient(window, "x", 2.0, np.random.default_rng(1), False)
    b = od.anonymise_patient(window, "x", 2.0, np.random.default_rng(1), False)
    assert a.equals(b)
    err = a["glucose_mgdl"].to_numpy() - window["glycemie"].to_numpy()
    assert abs(err.mean()) < 0.5 and 1.0 < err.std() < 3.0
    assert a["glucose_mgdl"].between(od.GLUCOSE_MIN, od.GLUCOSE_MAX).all()


def test_phase_shift_keeps_structure_but_moves_start(window):
    out = od.anonymise_patient(window, "x", 2.0, np.random.default_rng(3), phase_shift=True)
    assert 0 <= out["time_h"].iloc[0] < 24
    assert np.allclose(np.diff(out["time_h"]), 5 / 60, atol=1e-3)


def test_mmoll_release_matches_mgdl_release(window):
    mg = od.anonymise_patient(window, "x", 2.0, np.random.default_rng(5), False)
    mm = od.anonymise_patient(window, "x", 2.0, np.random.default_rng(5), False, unit="mmol/L")
    assert "glucose_mmoll" in mm.columns
    assert np.allclose(mm["glucose_mmoll"] * to_raw.MGDL_PER_MMOLL, mg["glucose_mgdl"], atol=0.5)


@pytest.mark.parametrize("column,factor", [("glucose_mgdl", 1.0), ("glucose_mmoll", 1 / to_raw.MGDL_PER_MMOLL)])
def test_open_data_to_raw_layout(tmp_path, monkeypatch, column, factor):
    csv = tmp_path / "cohort_open_data.csv"
    pd.DataFrame({"patient_id": ["OD_001"] * 3 + ["OD_002"] * 2,
                  "time_h": [0, 5 / 60, 10 / 60, 0, 5 / 60],
                  column: np.array([100, 110, 120, 90, 95]) * factor}).to_csv(csv, index=False)
    raw = tmp_path / "Data" / "Raw"
    monkeypatch.setattr(to_raw, "RAW_PATH", raw)
    monkeypatch.setattr("sys.argv", ["open_data_to_raw", "--input", str(csv)])
    to_raw.main()

    df = pd.read_excel(raw / "OD_001.xlsx")
    assert list(df.columns) == ["time", "glycemie"]
    assert df["glycemie"].tolist() == pytest.approx([100, 110, 120], abs=0.1)
    assert df["time"].iloc[0] == to_raw.ORIGIN                       # synthetic origin, no real date
    assert (df["time"].diff().dropna() == pd.Timedelta(minutes=5)).all()
    assert (raw / "OD_002.xlsx").exists()


def test_sample_data_is_synthetic_reproducible_and_relative():
    prof = sample.PROFILES[0]
    a = sample.synth_patient(np.random.default_rng(sample.SEED), prof)
    b = sample.synth_patient(np.random.default_rng(sample.SEED), prof)
    assert a.equals(b)
    assert sample.START.year == 2000                                   # arbitrary t = 0, not a real date
    full = sample.DAYS * 24 * 60 // sample.STEP_MIN
    assert 0.9 * full < len(a) <= full                                 # full grid minus a few sensor gaps
