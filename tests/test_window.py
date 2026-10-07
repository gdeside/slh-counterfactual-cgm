"""45-day window extraction: long recordings, data that already are one window, too-short recordings."""
import pandas as pd

from src.config import WINDOW_DAYS
from src.preprocessing.window import find_optimal_window

STEP = 5.0


def _trace(days, step_min=STEP):
    n = int(days * 24 * 60 / step_min)
    t = pd.date_range("2000-01-01", periods=n, freq=f"{int(step_min)}min")
    return pd.DataFrame({"time": t, "glycemie": 120.0, "flag_time_error": False, "flag_outlier_glucose": False})


def test_longer_recording_gives_a_window():
    res = find_optimal_window(_trace(WINDOW_DAYS + 5), STEP)
    assert res is not None and res["coverage_pct"] > 95


def test_data_that_already_are_one_window_are_accepted():
    """The open dataset spans 45 days minus one sample; it must still yield its window."""
    res = find_optimal_window(_trace(WINDOW_DAYS), STEP)            # 45 d minus 5 min of span
    assert res is not None and res["coverage_pct"] > 95


def test_recording_clearly_shorter_than_the_window_is_rejected():
    assert find_optimal_window(_trace(WINDOW_DAYS - 1), STEP) is None
    assert find_optimal_window(_trace(33), STEP) is None            # 73 % of the window would otherwise pass
