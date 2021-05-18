import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import Undersampler


def test_init():
    parameters = {
        "sampling_ratio": 1,
        "min_samples": 1,
        "min_percentage": 0.5
    }
    undersampler = Undersampler(**parameters)
    assert undersampler.parameters == parameters


def test_none_y():
    X = pd.DataFrame([[i] for i in range(5)])
    undersampler = Undersampler()
    with pytest.raises(ValueError, match="y cannot be none"):
        undersampler.fit(X, None)
    with pytest.raises(ValueError, match="y cannot be none"):
        undersampler.fit_transform(X, None)
    undersampler.fit(X, pd.Series([0] * 4 + [1]))
    undersampler.transform(X, None)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_no_undersample(data_type, make_data_type, X_y_binary):
    X, y = X_y_binary
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    undersampler = Undersampler()
    new_X, new_y = undersampler.fit_transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, new_X.to_dataframe().values)
    np.testing.assert_equal(y, new_y.to_series().values)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_undersample_imbalanced(data_type, make_data_type):
    X = np.array([[i] for i in range(1000)])
    y = np.array([0] * 150 + [1] * 850)
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    sampling_ratio = 0.25
    undersampler = Undersampler(sampling_ratio=sampling_ratio)
    new_X, new_y = undersampler.fit_transform(X, y)

    assert len(new_X) == 750
    assert len(new_y) == 750
    value_counts = new_y.to_series().value_counts()
    assert value_counts.values[1] / value_counts.values[0] == sampling_ratio
    pd.testing.assert_series_equal(value_counts, pd.Series([600, 150], index=[1, 0]), check_dtype=False)

    transform_X, transform_y = undersampler.transform(X, y)

    if data_type == "ww":
        X = X.to_dataframe().values
        y = y.to_series().values
    elif data_type == "pd":
        X = X.values
        y = y.values

    np.testing.assert_equal(X, transform_X.to_dataframe().values)
    np.testing.assert_equal(None, transform_y)
