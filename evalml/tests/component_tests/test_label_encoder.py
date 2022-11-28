import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from evalml.pipelines.components import LabelEncoder


def test_label_encoder_init():
    encoder = LabelEncoder()
    assert encoder.parameters == {"positive_label": None}
    assert encoder.random_seed == 0


def test_label_encoder_fit_transform_y_is_None():
    X = pd.DataFrame({})
    y = pd.Series(["a", "b"])
    encoder = LabelEncoder()
    with pytest.raises(ValueError, match="y cannot be None"):
        encoder.fit(X)

    encoder.fit(X, y)
    with pytest.raises(ValueError, match="y cannot be None"):
        encoder.inverse_transform(None)


def test_label_encoder_transform_y_is_None():
    X = pd.DataFrame({})
    y = pd.Series(["a", "b"])
    encoder = LabelEncoder()
    encoder.fit(X, y)
    X_t, y_t = encoder.transform(X)
    assert_frame_equal(X, X_t)
    assert y_t is None


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


def test_label_encoder_with_positive_label_multiclass_error():
    y = pd.Series(["a", "b", "c", "a"])
    encoder = LabelEncoder(positive_label="a")
    with pytest.raises(
        ValueError,
        match="positive_label should only be set for binary classification targets",
    ):
        encoder.fit(None, y)


def test_label_encoder_with_positive_label_missing_from_input():
    y = pd.Series(["a", "b", "a"])
    encoder = LabelEncoder(positive_label="z")
    with pytest.raises(
        ValueError,
        match="positive_label was set to `z` but was not found in the input target data.",
    ):
        encoder.fit(None, y)


@pytest.mark.parametrize(
    "y, positive_label, y_encoded_expected",
    [
        (
            pd.Series([True, False, False, True]),
            False,
            pd.Series([0, 1, 1, 0]),
        ),  # boolean
        (
            pd.Series([True, False, False, True]),
            True,
            pd.Series([1, 0, 0, 1]),
        ),  # boolean
        (
            pd.Series([0, 1, 1, 0]),
            0,
            pd.Series([1, 0, 0, 1]),
        ),  # int, 0 / 1, encoding should flip
        (
            pd.Series([0, 1, 1, 0]),
            1,
            pd.Series([0, 1, 1, 0]),
        ),  # int, 0 / 1, encoding should not change
        (
            pd.Series([6, 2, 2, 6]),
            6,
            pd.Series([1, 0, 0, 1]),
        ),  # ints, not 0 / 1, encoding should not change
        (
            pd.Series([6, 2, 2, 6]),
            2,
            pd.Series([0, 1, 1, 0]),
        ),  # ints, not 0 / 1, encoding should flip
        (pd.Series(["b", "a", "a", "b"]), "a", pd.Series([0, 1, 1, 0])),  # categorical
        (pd.Series(["b", "a", "a", "b"]), "b", pd.Series([1, 0, 0, 1])),  # categorical
    ],
)
def test_label_encoder_with_positive_label(y, positive_label, y_encoded_expected):
    encoder = LabelEncoder(positive_label=positive_label)

    _, y_fit_transformed = encoder.fit_transform(None, y)
    assert_series_equal(y_encoded_expected, y_fit_transformed)

    y_inverse_transformed = encoder.inverse_transform(y_fit_transformed)
    assert_series_equal(ww.init_series(y), y_inverse_transformed)


def test_label_encoder_with_positive_label_fit_different_from_transform():
    encoder = LabelEncoder(positive_label="a")
    y = pd.Series(["a", "b", "b", "a"])
    encoder.fit(None, y)
    with pytest.raises(ValueError, match="y contains previously unseen labels"):
        encoder.transform(None, pd.Series(["x", "y", "x"]))


@pytest.mark.parametrize("use_positive_label", [True, False])
def test_label_encoder_transform_does_not_have_all_labels(use_positive_label):
    encoder = LabelEncoder(positive_label="a" if use_positive_label else None)
    y = pd.Series(["a", "b", "b", "a"])
    encoder.fit(None, y)
    expected = (
        pd.Series([1, 1, 1, 1]) if use_positive_label else pd.Series([0, 0, 0, 0])
    )
    _, y_transformed = encoder.transform(None, pd.Series(["a", "a", "a", "a"]))
    assert_series_equal(expected, y_transformed)


def test_label_encoder_with_positive_label_with_custom_indices():
    encoder = LabelEncoder(positive_label="a")
    y = pd.Series(["a", "b", "a"])
    encoder.fit(None, y)
    y_with_custom_indices = pd.Series(["b", "a", "a"], index=[5, 6, 7])
    _, y_transformed = encoder.transform(None, y_with_custom_indices)
    assert_index_equal(y_with_custom_indices.index, y_transformed.index)
