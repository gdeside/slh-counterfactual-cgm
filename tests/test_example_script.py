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


# ---- ShanghaiT1DM converter (scripts/examples/convert_shanghai_t1dm.py) ---------------------------------------
def _shanghai_sheet(unit="mg / dl", n=200, gap=None):
    from scripts.examples import convert_shanghai_t1dm as sh  # noqa: F401  (import check)
    t = pd.date_range("2021-07-30 16:43", periods=n, freq="15min")
    g = 150 + 30 * np.sin(np.arange(n) / 8)
    sheet = pd.DataFrame({"Date": t, f"CGM ({unit})": g, "Insulin dose - s.c.": np.nan})
    return sheet.drop(index=range(*gap)) if gap else sheet


def test_shanghai_conversion_5min_mgdl():
    from scripts.examples import convert_shanghai_t1dm as sh
    out = sh.to_project_format(_shanghai_sheet())
    assert list(out.columns) == ["time", "glycemie"]
    assert out["time"].diff().dropna().eq(pd.Timedelta("5min")).all()
    assert len(out) == 199 * 3 + 1                                   # 15 min -> 5 min, original points kept


def test_shanghai_conversion_mmoll_and_long_gap_not_filled():
    from scripts.examples import convert_shanghai_t1dm as sh
    mg = sh.to_project_format(_shanghai_sheet("mg / dl"))
    mm = _shanghai_sheet("mmol / L")
    mm["CGM (mmol / L)"] = mm["CGM (mmol / L)"] / MGDL_PER_MMOLL
    assert np.allclose(sh.to_project_format(mm)["glycemie"], mg["glycemie"], atol=0.1)
    gapped = sh.to_project_format(_shanghai_sheet(gap=(50, 60)))     # 2.5 h hole between real readings
    assert gapped["time"].diff().max() > pd.Timedelta("2h")


def test_shanghai_folder_skips_summary_and_writes_notice(tmp_path):
    from scripts.examples import convert_shanghai_t1dm as sh
    raw, out = tmp_path / "raw", tmp_path / "converted"
    raw.mkdir()
    _shanghai_sheet().to_excel(raw / "1001_0_20210730.xlsx", index=False)
    pd.DataFrame({"Patient Number": ["1001_0_20210730"]}).to_excel(raw / "Shanghai_T1DM_Summary.xlsx", index=False)
    assert sh.convert_folder(raw, out) == 1
    assert (out / "1001_0_20210730.csv").exists() and (out / "MODIFIED_DATA_NOTICE.md").exists()
    assert not (out / "Shanghai_T1DM_Summary.csv").exists()
