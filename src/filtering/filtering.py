
from src.filtering.filtering_config import MEDIAN_GLUCOSE_MIN, MEDIAN_GLUCOSE_MAX, COL_MEDIAN_GLUCOSE


def apply_median_glucose_filter(df):
    """
    Filter patients by median glucose criterion.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient metrics dataframe
        
    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only selected patients
    """
    mask = (df[COL_MEDIAN_GLUCOSE] >= MEDIAN_GLUCOSE_MIN) & \
           (df[COL_MEDIAN_GLUCOSE] <= MEDIAN_GLUCOSE_MAX)
    
    return df[mask].copy().reset_index(drop=True)