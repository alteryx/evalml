import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from evalml.pipelines.components import PolynomialDetrender

pytest.importorskip('sktime', reason='Skipping polynomial detrending tests because sktime not installed')


def test_polynomial_detrender_init():
    delayed_features = PolynomialDetrender(degree=3)
    assert delayed_features.parameters == {"degree": 3}


def test_polynomial_detrender_init_raises_error_if_degree_not_int():

    with pytest.raises(TypeError, match="Received str"):
        PolynomialDetrender(degree="1")

    with pytest.raises(TypeError, match="Received float"):
        PolynomialDetrender(degree=3.4)

    _ = PolynomialDetrender(degree=3.0)


def test_polynomial_detrender_raises_value_error_target_is_none(ts_data):
    X, y = ts_data

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDetrender!"):
        PolynomialDetrender(degree=3).fit_transform(X, None)

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDetrender!"):
        PolynomialDetrender(degree=3).fit(X, None)

    pdt = PolynomialDetrender(degree=3).fit(X, y)

    with pytest.raises(ValueError, match="y cannot be None for PolynomialDetrender!"):
        pdt.inverse_transform(X, None)


@pytest.mark.parametrize("input_type", ['np', 'pd', 'ww'])
@pytest.mark.parametrize("use_int_index", [True, False])
@pytest.mark.parametrize("degree", [1, 2, 3])
def test_polynomial_detrender_fit_transform(degree, use_int_index, input_type, ts_data):

    X_input, y_input = ts_data

    if use_int_index:
        X_input.index = np.arange(X_input.shape[0])
        y_input.index = np.arange(y_input.shape[0])

    # Get the expected answer
    lin_reg = LinearRegression(fit_intercept=True)
    features = PolynomialFeatures(degree=degree).fit_transform(np.arange(X_input.shape[0]).reshape(-1, 1))
    lin_reg.fit(features, y_input)
    detrended_values = y_input.values - lin_reg.predict(features)
    expected_index = y_input.index if input_type != 'np' else range(y_input.shape[0])
    expected_answer = pd.Series(detrended_values, index=expected_index)

    X, y = X_input, y_input

    if input_type == 'np':
        X = X_input.values
        y = y_input.values
    elif input_type == 'ww':
        X = ww.DataTable(X_input)
        y = ww.DataColumn(y_input)

    output_X, output_y = PolynomialDetrender(degree=degree).fit_transform(X, y)
    pd.testing.assert_series_equal(expected_answer, output_y.to_series())

    # Verify the X is not changed
    if input_type == "np":
        np.testing.assert_equal(X, output_X)
    elif input_type == "ww":
        pd.testing.assert_frame_equal(X.to_dataframe(), output_X.to_dataframe())
    else:
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


def test_polynomial_detrender_needs_monotonic_index(ts_data):
    X, y = ts_data
    detrender = PolynomialDetrender(degree=2)

    with pytest.raises(ValueError, match="The \\(time\\) index must be sorted \\(monotonically increasing\\)"):
        y_shuffled = y.sample(frac=1, replace=False)
        detrender.fit_transform(X, y_shuffled)

    with pytest.raises(NotImplementedError, match="class 'pandas.core.indexes.base.Index'> is not supported"):
        y_string_index = pd.Series(np.arange(31), index=[f"row_{i}" for i in range(31)])
        detrender.fit_transform(X, y_string_index)
