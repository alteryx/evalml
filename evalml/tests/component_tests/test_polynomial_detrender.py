import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import PolynomialDetrender

pytest.importorskip('sktime', reason='Skipping polynomial detrending tests because sktime not installed')


def test_polynomial_detrender_init():
    delayed_features = PolynomialDetrender(degree=3)
    assert delayed_features.parameters == {"degree": 3}


@pytest.mark.parametrize("use_int_index", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_detrender_fit_transform(degree, use_int_index, ts_data):
    X, y = ts_data
    if use_int_index:
        X.index = np.arange(X.shape[0])
        y.index = np.arange(y.shape[0])

    lin_reg = LinearRegression(fit_intercept=True)
    features = PolynomialFeatures(degree=degree).fit_transform(np.arange(X.shape[0]).reshape(-1, 1))
    lin_reg.fit(features, y)
    expected_values = pd.Series(y.values - lin_reg.predict(features), index=y.index)

    output_X, output_y = PolynomialDetrender(degree=degree).fit_transform(X, y)
    pd.testing.assert_series_equal(expected_values, output_y.to_series())
    pd.testing.assert_frame_equal(X, output_X)


@pytest.mark.parametrize("use_int_index", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_detrender_inverse_transform(degree, use_int_index, ts_data):
    X, y = ts_data
    if use_int_index:
        X.index = np.arange(X.shape[0])
        y.index = np.arange(y.shape[0])

    detrender = PolynomialDetrender(degree=degree)
    output_X, output_y = detrender.fit_transform(X, y)
    output_inverse_X, output_inverse_y = detrender.inverse_transform(output_X, output_y)
    pd.testing.assert_series_equal(y, output_inverse_y.to_series(), check_dtype=False)
    pd.testing.assert_frame_equal(X, output_inverse_X)
