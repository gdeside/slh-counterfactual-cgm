"""mg/dL <-> mmol/L: the pipeline computes in mg/dL, mmol/L is only a presentation layer."""
import numpy as np
import pandas as pd
import pytest

from src.utils.unit_conversion import (
    MGDL_PER_MMOLL, UNIT_MGDL, UNIT_MMOLL, mgdl_to_mmoll, mmoll_to_mgdl, normalize_unit,
)


def test_known_values():
    assert float(mgdl_to_mmoll(180, decimals=1)) == 10.0
    assert float(mgdl_to_mmoll(70, decimals=2)) == 3.88


def test_round_trip_is_lossless():
    x = np.array([40.0, 70.0, 126.5, 180.0, 400.0])
    assert np.allclose(mmoll_to_mgdl(mgdl_to_mmoll(x)), x)


def test_series_and_nan_are_preserved():
    s = pd.Series([90.0, np.nan, 180.0])
    out = mgdl_to_mmoll(s)
    assert isinstance(out, pd.Series) and np.isnan(out.iloc[1])


def test_conversion_factor_is_consistent():
    assert mmoll_to_mgdl(1.0) == pytest.approx(MGDL_PER_MMOLL)


@pytest.mark.parametrize("raw,expected", [("mg/dL", UNIT_MGDL), ("mgdl", UNIT_MGDL),
                                          ("mmol/L", UNIT_MMOLL), ("mmoll", UNIT_MMOLL)])
def test_normalize_unit(raw, expected):
    assert normalize_unit(raw) == expected


def test_normalize_unit_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_unit("g/L")
