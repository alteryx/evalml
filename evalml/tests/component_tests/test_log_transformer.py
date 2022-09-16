import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import LogTransformer


def test_log_transformer_init():
    log_ = LogTransformer()
    assert log_.parameters == {}


def test_log_transformer_no_y(X_y_regression):
    X, y = X_y_regression
    y = None

    output_X, output_y = LogTransformer().fit_transform(X, y)
    pd.testing.assert_frame_equal(X, output_X)
    assert not output_y


@pytest.mark.parametrize("input_type", ["np", "pd", "ww"])
@pytest.mark.parametrize("data_type", ["positive", "mixed", "negative"])
def test_log_transformer_fit_transform(data_type, input_type, X_y_regression):
    X, y = X_y_regression
    if data_type == "positive":
        y = np.abs(y)
    elif data_type == "negative":
        y = -np.abs(y)

    if input_type == "np":
        X = X.values
        y = y.values
    elif input_type == "pd":
        X = pd.DataFrame(X)
        y = pd.Series(y)

    if y.min() <= 0:
        y = y + abs(y.min()) + 1
    expected_log = np.log(y)

    output_X, output_y = LogTransformer().fit_transform(X, y)

    pd.testing.assert_series_equal(pd.Series(expected_log), output_y)

    # Verify the X is not changed
    if input_type == "np":
        np.testing.assert_equal(X, output_X)
    else:
        pd.testing.assert_frame_equal(X, output_X)


@pytest.mark.parametrize("is_time_series", [True, False])
@pytest.mark.parametrize("data_type", ["positive", "mixed", "negative"])
def test_log_transformer_inverse_transform(
    data_type,
    is_time_series,
    X_y_regression,
    ts_data,
):
    if is_time_series:
        X, _, y = ts_data()
    else:
        X, y = X_y_regression

    if data_type == "positive":
        y = np.abs(y)
    elif data_type == "negative":
        y = -np.abs(y)

    log_ = LogTransformer()
    output_X, output_y = log_.fit_transform(X, y)
    output_inverse_y = log_.inverse_transform(output_y)
    pd.testing.assert_series_equal(pd.Series(y), output_inverse_y, check_dtype=False)
