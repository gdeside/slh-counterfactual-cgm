"""BrisT1D-Open converter: glucose only, mg/dL, 5-min grid."""
import pandas as pd
import pytest

from scripts.examples.convert_brist1d import convert_file


def test_keeps_only_glucose_in_mgdl_on_5min_grid(tmp_path):
    f = tmp_path / "P01.csv"
    pd.DataFrame({
        "timestamp": ["2023-06-01 00:05:00", "2023-06-01 00:06:00", "2023-06-01 00:10:00",
                      "2023-06-01 00:14:00", "2023-06-01 00:16:00"],
        "bg": [None, 10.0, None, 10.2, 10.4],           # mmol/L; rows without glucose are insulin-only rows
        "insulin": [0.0083, None, 0.0083, None, None],
        "device": ["pump", "cgm", "pump", "cgm", "cgm"],
    }).to_csv(f, index=False)

    out = convert_file(f)

    assert list(out.columns) == ["time", "glycemie"]
    assert list(out["time"]) == list(pd.to_datetime(["2023-06-01 00:05", "2023-06-01 00:15"]))
    assert out["glycemie"].iloc[0] == pytest.approx(180.2, abs=0.05)                 # 10.0 mmol/L
    assert out["glycemie"].iloc[1] == pytest.approx(10.3 * 18.0182, abs=0.05)         # 00:14 and 00:16 averaged
