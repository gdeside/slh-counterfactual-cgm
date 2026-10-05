"""Shared fixtures: a small synthetic CGM trace with one episode of each kind."""
import numpy as np
import pandas as pd
import pytest

SPIKE = [150, 200, 230, 220, 190, 150]            # ~20 min above 180 mg/dL


@pytest.fixture
def synthetic_trace() -> pd.DataFrame:
    """3 days, 5-min sampling, baseline 120 mg/dL (arbitrary t = 0 = 2000-01-01), with:
      * day 1, 23:00   short spike at night            -> fails C3 (not SLH)
      * day 2, 12:00   short isolated daytime spike    -> SLH
      * day 3, 14:00   150-min excursion               -> fails C1 (LLH)
    """
    idx = pd.date_range("2000-01-01", periods=3 * 288, freq="5min")
    g = np.full(len(idx), 120.0)

    def put(day, hour, values):
        i = day * 288 + int(hour * 12)
        g[i:i + len(values)] = values

    put(0, 23, SPIKE)
    put(1, 12, SPIKE)
    put(2, 14, [150] + [220] * 30 + [150])

    return pd.DataFrame({
        "time": idx, "glycemie": g,
        "flag_time_error": False, "flag_gap_before": False,
        "gap_before_min": np.nan, "flag_outlier_glucose": False,
    })
