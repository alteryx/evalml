from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from pytest import importorskip
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
)

from evalml.exceptions import ComponentNotYetFittedError
from evalml.pipelines.components import TargetEncoder

importorskip(
    "category_encoders", reason="Skipping test because category_encoders not installed"
)


def test_init():
    parameters = {
        "cols": None,
        "smoothing": 1.0,
        "handle_unknown": "value",
        "handle_missing": "value",
    }
    encoder = TargetEncoder()
    assert encoder.parameters == parameters


def test_parameters():
    encoder = TargetEncoder(cols=["a"])
    expected_parameters = {
        "cols": ["a"],
        "smoothing": 1.0,
        "handle_unknown": "value",
        "handle_missing": "value",
    }
    assert encoder.parameters == expected_parameters


def test_categories():
    encoder = TargetEncoder()
    with pytest.raises(AttributeError, match="'TargetEncoder' object has no attribute"):
        encoder.categories


def test_invalid_inputs():
    with pytest.raises(ValueError, match="Invalid input 'test' for handle_unknown"):
        TargetEncoder(handle_unknown="test")
    with pytest.raises(ValueError, match="Invalid input 'test2' for handle_missing"):
        TargetEncoder(handle_missing="test2")
    with pytest.raises(
        ValueError, match="Smoothing value needs to be strictly larger than 0"
    ):
        TargetEncoder(smoothing=0)


def test_null_values_in_dataframe():
    X = pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d", np.nan],
            "col_2": ["a", "b", "a", "c", "b"],
            "col_3": ["a", "a", "a", "a", "a"],
        }
    )
    X.ww.init(
        logical_types={
            "col_1": "categorical",
            "col_2": "categorical",
            "col_3": "categorical",
        }
    )
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder(handle_missing="value")
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": [0.6, 0.6, 0.6, 0.6, 0.6],
            "col_2": [0.526894, 0.526894, 0.526894, 0.6, 0.526894],
            "col_3": [
                0.6,
                0.6,
                0.6,
                0.6,
                0.6,
            ],
        }
    )

    assert_frame_equal(X_expected, X_t)

    encoder = TargetEncoder(handle_missing="return_nan")
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": [0.6, 0.6, 0.6, 0.6, np.nan],
            "col_2": [0.526894, 0.526894, 0.526894, 0.6, 0.526894],
            "col_3": [
                0.6,
                0.6,
                0.6,
                0.6,
                0.6,
            ],
        }
    )
    assert_frame_equal(X_expected, X_t)

    encoder = TargetEncoder(handle_missing="error")
    with pytest.raises(ValueError, match="Columns to be encoded can not contain null"):
        encoder.fit(X, y)


def test_cols():
    X = pd.DataFrame(
        {
            "col_1": [1, 2, 1, 1, 2] * 2,
            "col_2": ["2", "1", "1", "1", "1"] * 2,
            "col_3": ["a", "a", "a", "a", "a"] * 2,
        }
    )
    X_expected = X.astype({"col_1": "int64", "col_2": "category", "col_3": "category"})
    y = pd.Series([0, 1, 1, 1, 0] * 2)
    encoder = TargetEncoder(cols=[])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    assert_frame_equal(X_expected, X_t)

    encoder = TargetEncoder(cols=["col_2"])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": pd.Series([1, 2, 1, 1, 2] * 2, dtype="int64"),
            "col_2": [0.161365, 0.749863, 0.749863, 0.749863, 0.749863] * 2,
            "col_3": pd.Series(["a", "a", "a", "a", "a"] * 2, dtype="category"),
        }
    )
    assert_frame_equal(X_expected, X_t, check_less_precise=True)

    encoder = TargetEncoder(cols=["col_2", "col_3"])
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    encoder2 = TargetEncoder()
    encoder2.fit(X, y)
    X_t2 = encoder2.transform(X)
    assert_frame_equal(X_t, X_t2)


def test_transform():
    X = pd.DataFrame(
        {
            "col_1": [1, 2, 1, 1, 2],
            "col_2": ["r", "t", "s", "t", "t"],
            "col_3": ["a", "a", "a", "b", "a"],
        }
    )
    X.ww.init(logical_types={"col_2": "categorical", "col_3": "categorical"})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder()
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": pd.Series([1, 2, 1, 1, 2], dtype="int64"),
            "col_2": [0.6, 0.65872, 0.6, 0.65872, 0.65872],
            "col_3": [0.504743, 0.504743, 0.504743, 0.6, 0.504743],
        }
    )
    assert_frame_equal(X_expected, X_t)


def test_smoothing():
    # larger smoothing values should bring the values closer to the global mean
    X = pd.DataFrame(
        {
            "col_1": [1, 2, 1, 1, 2],
            "col_2": [2, 1, 1, 1, 1],
            "col_3": ["a", "a", "a", "a", "b"],
        }
    )
    X.ww.init(logical_types={"col_3": "categorical"})
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder(smoothing=1)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": pd.Series([1, 2, 1, 1, 2], dtype="int64"),
            "col_2": pd.Series([2, 1, 1, 1, 1], dtype="int64"),
            "col_3": [0.742886, 0.742886, 0.742886, 0.742886, 0.6],
        }
    )
    assert_frame_equal(X_expected, X_t)

    encoder = TargetEncoder(smoothing=10)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": pd.Series([1, 2, 1, 1, 2], dtype="int64"),
            "col_2": pd.Series([2, 1, 1, 1, 1], dtype="int64"),
            "col_3": [0.686166, 0.686166, 0.686166, 0.686166, 0.6],
        }
    )
    assert_frame_equal(X_expected, X_t)

    encoder = TargetEncoder(smoothing=100)
    encoder.fit(X, y)
    X_t = encoder.transform(X)
    X_expected = pd.DataFrame(
        {
            "col_1": pd.Series([1, 2, 1, 1, 2], dtype="int64"),
            "col_2": pd.Series([2, 1, 1, 1, 1], dtype="int64"),
            "col_3": [0.676125, 0.676125, 0.676125, 0.676125, 0.6],
        }
    )
    assert_frame_equal(X_expected, X_t)


def test_get_feature_names():
    X = pd.DataFrame(
        {
            "col_1": [1, 2, 1, 1, 2],
            "col_2": ["r", "t", "s", "t", "t"],
            "col_3": ["a", "a", "a", "b", "a"],
        }
    )
    y = pd.Series([0, 1, 1, 1, 0])
    encoder = TargetEncoder()
    with pytest.raises(
        ComponentNotYetFittedError,
        match="This TargetEncoder is not fitted yet. You must fit",
    ):
        encoder.get_feature_names()
    encoder.fit(X, y)
    np.testing.assert_array_equal(
        encoder.get_feature_names(), np.array(["col_1", "col_2", "col_3"])
    )


@patch("evalml.pipelines.components.transformers.transformer.Transformer.fit")
def test_pandas_numpy(mock_fit, X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X).sample(frac=1)

    encoder = TargetEncoder()

    encoder.fit(X, y)
    assert_frame_equal(mock_fit.call_args[0][0], X)

    X_numpy = X.to_numpy()
    encoder.fit(X_numpy, y)


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(
            pd.to_datetime(["20190902", "20200519", "20190607"], format="%Y%m%d")
        ),
        pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 3.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"], dtype="category")),
        pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
        pd.DataFrame(
            pd.Series(
                ["this will be a natural language column because length", "yay", "hay"],
                dtype="string",
            )
        ),
    ],
)
def test_target_encoder_woodwork_custom_overrides_returned_by_components(X_df):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean, Datetime]
    for logical_type in override_types:
        try:
            X = X_df.copy()
            X.ww.init(logical_types={0: logical_type})
        except (ww.exceptions.TypeConversionError, ValueError, TypeError):
            continue

        encoder = TargetEncoder()
        encoder.fit(X, y)
        transformed = encoder.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)

        if logical_type == Categorical:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: Double
            }
        else:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: logical_type
            }
