import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.preprocessing.utils import drop_nan_target_rows


@pytest.fixture
def X_y_na():
    y = pd.Series([1, 0, 1, np.nan])
    X = pd.DataFrame()
    X["a"] = ['a', 'b', 'c', 'd']
    X["b"] = [1, 2, 3, 0]
    X["c"] = [np.nan, 0, 0, np.nan]
    X["d"] = [0, 0, 0, 0]
    return X, y


def test_drop_nan_target_rows(X_y_na):
    X, y = X_y_na
    X_t, y_t = drop_nan_target_rows(X, y)
    y_expected = pd.Series([1, 0, 1])
    X_expected = X.iloc[:-1]
    assert_frame_equal(X_expected, X_t, check_dtype=False)
    assert_series_equal(y_expected, y_t, check_dtype=False)


def test_with_numpy_input(X_y_na):
    _, y = X_y_na
    X_arr = np.array([[1, 2, 3, 0],
                      [np.nan, 0, 0, 1],
                      [np.nan, 0, np.nan, 0],
                      [np.nan, 0, 0, 0]])
    y_arr = y.values
    X_t, y_t = drop_nan_target_rows(X_arr, y_arr)
    y_expected = pd.Series([1, 0, 1])
    X_expected = pd.DataFrame(X_arr).iloc[:-1]
    assert_frame_equal(X_expected, X_t, check_dtype=False)
    assert_series_equal(y_expected, y_t, check_dtype=False)
