"""scripts/examples/detect_and_counterfactual.py: the two core steps on one standalone file."""
import numpy as np
import pandas as pd
import pytest

from scripts.examples import detect_and_counterfactual as ex
from src.utils.unit_conversion import MGDL_PER_MMOLL


def test_demo_runs_and_writes_outputs(tmp_path):
    ex.main(["--demo", "--out_dir", str(tmp_path)])
    for name in ("episodes.csv", "counterfactual.csv", "summary.txt", "counterfactual.png"):
        assert (tmp_path / name).stat().st_size > 0
    cf = pd.read_csv(tmp_path / "counterfactual.csv")
    assert cf["counterfactual"].max() <= cf["observed"].max()          # SLH peaks are removed, not added
    assert (cf["counterfactual"] < cf["observed"]).any()
    assert (cf["observed"] > 180).sum() > (cf["counterfactual"] > 180).sum()


def test_csv_with_other_column_names_and_mmoll(tmp_path):
    demo = ex.demo_trace()
    path = tmp_path / "patient.csv"
    pd.DataFrame({"Timestamp": demo["time"].dt.strftime("%Y-%m-%d %H:%M"),
                  "Glucose": demo["glycemie"] / MGDL_PER_MMOLL}).to_csv(path, sep=";", index=False)
    loaded = ex.load_trace(path, unit="mmol/L")
    assert np.allclose(loaded["glycemie"], demo["glycemie"], atol=0.01)
    table, cf, timestep = ex.run(loaded)
    assert timestep == 5.0 and (table["classification"] == "SLH").sum() == 2
    assert table["category"].value_counts().to_dict() == {"SLH": 2, "Night": 1, "LLH": 1}   # Figure 2 categories


def test_excel_input(tmp_path):
    demo = ex.demo_trace()
    path = tmp_path / "patient.xlsx"
    demo.to_excel(path, index=False)
    assert len(ex.load_trace(path)) == len(demo)


def test_unknown_columns_give_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"a": range(200), "b": range(200)}).to_csv(path, index=False)
    with pytest.raises(ex.TraceError, match="time column"):
        ex.load_trace(path)
    with pytest.raises(SystemExit, match="time column"):                  # command line: message, no traceback
        ex.main(["--input", str(path), "--out_dir", str(tmp_path / "o")])


# ---------------------------------------------------------------- folder mode (a whole cohort, no Data/Raw layout)
@pytest.fixture
def cohort_folder(tmp_path):
    folder = tmp_path / "my_cohort"
    folder.mkdir()
    demo = ex.demo_trace()
    demo.to_csv(folder / "P001.csv", index=False)
    demo.rename(columns={"time": "Timestamp", "glycemie": "Glucose"}).to_csv(folder / "P002.csv", sep=";", index=False)
    demo.to_excel(folder / "P003.xlsx", index=False)
    pd.DataFrame({"a": range(200), "b": range(200)}).to_csv(folder / "bad_columns.csv", index=False)   # unreadable layout
    demo.head(50).to_csv(folder / "too_short.csv", index=False)
    (folder / ".hidden.csv").write_text("x")
    (folder / "~$P003.xlsx").write_bytes(b"lock")
    (folder / "notes.md").write_text("not a CGM file")
    return folder


def test_list_patient_files_ignores_non_cgm_files(cohort_folder):
    names = [p.name for p in ex.list_patient_files(cohort_folder)]
    assert names == ["P001.csv", "P002.csv", "P003.xlsx", "bad_columns.csv", "too_short.csv"]


def test_folder_mode_processes_every_patient_and_skips_bad_files(cohort_folder, tmp_path):
    out = tmp_path / "out"
    ex.main(["--input_dir", str(cohort_folder), "--out_dir", str(out)])
    summary = pd.read_csv(out / "cohort_summary.csv")
    assert summary["patient_id"].tolist() == ["P001", "P002", "P003"]            # file name = patient ID
    assert (summary["n_SLH"] == 2).all() and (summary["n_Night"] == 1).all() and (summary["n_LLH"] == 1).all()
    assert (summary["delta_TAR_180_pct"] < 0).all()                              # SLH removal lowers time above range
    episodes = pd.read_csv(out / "cohort_episodes.csv")
    assert episodes.columns[0] == "patient_id" and set(episodes["patient_id"]) == {"P001", "P002", "P003"}
    assert len(episodes) == 12
    skipped = pd.read_csv(out / "skipped_files.csv")
    assert sorted(skipped["file"]) == ["bad_columns.csv", "too_short.csv"]
    for pid in ("P001", "P002", "P003"):
        for name in ("episodes.csv", "counterfactual.csv", "counterfactual.png"):
            assert (out / pid / name).stat().st_size > 0


def test_folder_mode_no_plots(cohort_folder, tmp_path):
    out = tmp_path / "out"
    ex.main(["--input_dir", str(cohort_folder), "--out_dir", str(out), "--no_plots"])
    assert (out / "P001" / "episodes.csv").exists() and not (out / "P001" / "counterfactual.png").exists()


def test_folder_mode_mmoll_files(tmp_path):
    folder = tmp_path / "mmol"
    folder.mkdir()
    demo = ex.demo_trace()
    pd.DataFrame({"time": demo["time"], "glucose": demo["glycemie"] / MGDL_PER_MMOLL}).to_csv(folder / "Q1.csv", index=False)
    ex.main(["--input_dir", str(folder), "--unit", "mmol/L", "--out_dir", str(tmp_path / "o"), "--no_plots"])
    assert pd.read_csv(tmp_path / "o" / "cohort_summary.csv").loc[0, "n_SLH"] == 2


def test_empty_or_missing_folder_gives_clear_error(tmp_path):
    with pytest.raises(SystemExit, match="no CSV"):
        ex.main(["--input_dir", str(tmp_path), "--out_dir", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not a folder"):
        ex.main(["--input_dir", str(tmp_path / "missing"), "--out_dir", str(tmp_path / "o")])
