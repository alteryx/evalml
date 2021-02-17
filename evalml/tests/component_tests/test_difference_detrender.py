import pytest
from evalml.pipelines.components import DifferenceDetrender
import numpy as np
import pandas as pd


def test_polynomial_detrender_init():
    delayed_features = DifferenceDetrender(degree=3)
    assert delayed_features.degree == 3
    assert delayed_features.parameters == {"degree": 3}


@pytest.mark.parametrize("use_int_index", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_difference_detrender_fit_transform(degree, use_int_index, ts_data):
    X, y = ts_data
    X = X.iloc[:5]
    y = pd.Series([5, 7, 4, 6, 11], index=X.index)
    if use_int_index:
        X.index = np.arange(X.shape[0])
        y.index = np.arange(y.shape[0])

    expected_values = pd.Series(y)
    for degree in range(1, degree + 1):
        expected_values -= expected_values.shift(1)

    output_X, output_y = DifferenceDetrender(degree=degree).fit_transform(X, y)
    pd.testing.assert_series_equal(expected_values, output_y.to_series())
    pd.testing.assert_frame_equal(X, output_X)


@pytest.mark.parametrize("use_int_index", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_detrender_inverse_transform(degree, use_int_index, ts_data):
    X, y = ts_data
    X = X.iloc[:5]
    y = pd.Series([5, 7, 4, 6, 11], index=X.index)
    if use_int_index:
        X.index = np.arange(X.shape[0])
        y.index = np.arange(y.shape[0])

    detrender = DifferenceDetrender(degree=degree)
    output_X, output_y = detrender.fit_transform(X, y)
    output_inverse_X, output_inverse_y = detrender.inverse_transform(output_X, output_y)
    pd.testing.assert_series_equal(y, output_inverse_y.to_series(), check_dtype=False)
    pd.testing.assert_frame_equal(X, output_inverse_X)