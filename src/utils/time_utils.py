import pandas as pd
import numpy as np

from src.config import NIGHT_START_H, NIGHT_END_H

def is_night(timestamp: pd.Timestamp) -> bool:
    """Check if timestamp falls in night window [22:00, 06:00)."""
    return timestamp.hour >= NIGHT_START_H or timestamp.hour < NIGHT_END_H


def is_daytime_mask(time_series: pd.Series) -> np.ndarray:
    """Return boolean mask: True for daytime, False for night."""
    return ~time_series.apply(is_night).values
