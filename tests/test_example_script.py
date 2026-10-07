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
    with pytest.raises(SystemExit):
        ex.load_trace(path)
