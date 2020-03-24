import numpy as np
import pandas as pd
import pytest

from evalml.preprocessing.utils import drop_nan_rows


@pytest.fixture
def X_y_na():
    y = pd.Series([1, 0, 1, np.nan])
    X = pd.DataFrame()
    X["a"] = ['a', 'b', 'c', 'd']
    X["b"] = [1, 2, 3, 0]
    X["c"] = [np.nan, 0, 0, np.nan]
    X["d"] = [0, 0, 0, 0]
    return X, y


def test_drop_nan_rows(X_y_na):
    X, y = X_y_na
    X_t, y_t = drop_nan_rows(X, y)
    assert len(X_t) == 3
    assert len(y_t) == 3


def test_with_numpy_input(X_y_na):
    _, y = X_y_na
    X_arr = np.array([[1, 2, 3, 0],
                      [np.nan, 0, 0, 1],
                      [np.nan, 0, np.nan, 0],
                      [np.nan, 0, 0, 0]])
    y_arr = y.values
    X_t, y_t = drop_nan_rows(X_arr, y_arr)
    assert len(X_t) == 3
    assert len(y_t) == 3
