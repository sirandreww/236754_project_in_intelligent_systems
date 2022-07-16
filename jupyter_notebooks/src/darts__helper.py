"""
***********************************************************************************************************************
    imports
***********************************************************************************************************************
"""

import darts
import numpy as np


"""
***********************************************************************************************************************
    api functions
***********************************************************************************************************************
"""


def get_darts_series_from_df(df):
    arr = np.float32(df["sample"].to_numpy())
    res = darts.timeseries.TimeSeries.from_values(arr)
    return res


def get_np_array_from_series(series):
    res_np_arr = np.float64(series.pd_series().to_numpy())
    assert isinstance(res_np_arr, np.ndarray)
    assert res_np_arr.dtype == np.float64
    return res_np_arr
