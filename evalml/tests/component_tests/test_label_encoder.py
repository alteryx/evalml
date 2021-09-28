import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines.components import LabelEncoder


def test_label_encoder_init():
    encoder = LabelEncoder()
    assert encoder.parameters == {}
    assert encoder.random_seed == 0


def test_label_encoder_fit_transform_y_is_None():
    X = pd.DataFrame({})
    y = pd.Series(["a", "b"])
    encoder = LabelEncoder()
    with pytest.raises(ValueError, match="y cannot be None"):
        encoder.fit(X)

    encoder.fit(X, y)
    with pytest.raises(ValueError, match="y cannot be None"):
        encoder.transform(X)

    with pytest.raises(ValueError, match="y cannot be None"):
        encoder.inverse_transform(None)


def test_label_encoder_fit_transform_with_numeric_values_does_not_encode():
    X = pd.DataFrame({})
    # binary
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y, y_t)

    # multiclass
    X = pd.DataFrame({})
    y = pd.Series([0, 1, 1, 2, 0, 2])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y, y_t)


def test_label_encoder_fit_transform_with_numeric_values_needs_encoding():
    X = pd.DataFrame({})
    # binary
    y = pd.Series([2, 1, 2, 1])
    y_expected = pd.Series([1, 0, 1, 0])

    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y_expected, y_t)

    # multiclass
    y = pd.Series([0, 1, 1, 3, 0, 3])
    y_expected = pd.Series([0, 1, 1, 2, 0, 2])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y_expected, y_t)


def test_label_encoder_fit_transform_with_categorical_values():
    X = pd.DataFrame({})
    # binary
    y = pd.Series(["b", "a", "b", "b"])
    y_expected = pd.Series([1, 0, 1, 1])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y_expected, y_t)

    # multiclass
    y = pd.Series(["c", "a", "b", "c", "d"])
    y_expected = pd.Series([2, 0, 1, 2, 3])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X, y)
    assert_frame_equal(X, X_t)
    assert_series_equal(y_expected, y_t)


def test_label_encoder_fit_transform_equals_fit_and_transform():
    X = pd.DataFrame({})
    y = pd.Series(["a", "b", "c", "a"])

    encoder = LabelEncoder()
    X_fit_transformed, y_fit_transformed = encoder.fit_transform(X, y)

    encoder_duplicate = LabelEncoder()
    encoder_duplicate.fit(X, y)
    X_transformed, y_transformed = encoder_duplicate.transform(X, y)

    assert_frame_equal(X_fit_transformed, X_transformed)
    assert_series_equal(y_fit_transformed, y_transformed)


def test_label_encoder_inverse_transform():
    X = pd.DataFrame({})
    y = pd.Series(["a", "b", "c", "a"])
    y_expected = ww.init_series(y)
    encoder = LabelEncoder()
    _, y_fit_transformed = encoder.fit_transform(X, y)
    y_inverse_transformed = encoder.inverse_transform(y_fit_transformed)
    assert_series_equal(y_expected, y_inverse_transformed)

    y_encoded = pd.Series([1, 0, 2, 1])
    y_expected = ww.init_series(pd.Series(["b", "a", "c", "b"]))
    y_inverse_transformed = encoder.inverse_transform(y_encoded)
    assert_series_equal(y_expected, y_inverse_transformed)
