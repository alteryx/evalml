import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evalml.pipelines.components import TypedImputer


def test_invalid_strategy_parameters():
    with pytest.raises(ValueError, match="Valid numerical impute strategies are"):
        TypedImputer(numeric_impute_strategy="not a valid strategy")
    with pytest.raises(ValueError, match="Valid categorical impute strategies are"):
        TypedImputer(categorical_impute_strategy="mean")


def test_typed_imputer_init():
    imputer = TypedImputer(categorical_impute_strategy="most_frequent",
                           numeric_impute_strategy="median")
    expected_parameters = {
        'categorical_impute_strategy': 'most_frequent',
        'numeric_impute_strategy': 'median',
        'fill_value': None
    }
    expected_hyperparameters = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"]
    }
    assert imputer.name == "Typed Imputer"
    assert imputer.parameters == expected_parameters
    assert imputer.hyperparameter_ranges == expected_hyperparameters


def test_numeric_only_input():
    X = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [np.nan, 1, 2, 1, 0],
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]

    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TypedImputer(numeric_impute_strategy="median")
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [1, 1, 2, 1, 0],
        "float with nan": [0.0, 1.0, 0, -1.0, 0.]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TypedImputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_only_input():
    X = pd.DataFrame({
        "categorical col": pd.Series([0, 1, 2, 0, 3], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series([np.nan, 1, np.nan, 0, 3], dtype='category'),
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
        "all nan": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='category')
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TypedImputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "categorical col": pd.Series([0, 1, 2, 0, 3], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series([0, 1, 0, 0, 3], dtype='category'),
        "object with nan": ["b", "b", "b", "c", "b"],
        "bool col with nan": [True, True, False, True, True]
    })

    imputer = TypedImputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_and_numeric_input():
    X = pd.DataFrame({
        "categorical col": pd.Series([0, 1, 2, 0, 3], dtype='category'),
        "int col": [0, 1, 2, 0, 3],
        "object col": ["b", "b", "a", "c", "d"],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "bool col": [True, False, False, True, True],
        "int with nan": [np.nan, 1, 2, 1, 0],
        "categorical with nan": pd.Series([np.nan, 1, np.nan, 0, 3], dtype='category'),
        "float with nan": [0.0, 1.0, np.nan, -1.0, 0.],
        "object with nan": ["b", "b", np.nan, "c", np.nan],
        "bool col with nan": [True, np.nan, False, np.nan, True],
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TypedImputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame({
        "int col": [0, 1, 2, 0, 3],
        "float col": [0.0, 1.0, 0.0, -2.0, 5.],
        "int with nan": [1, 1, 2, 1, 0],
        "float with nan": [0.0, 1.0, 0, -1.0, 0.],
        "categorical col": pd.Series([0, 1, 2, 0, 3], dtype='category'),
        "object col": ["b", "b", "a", "c", "d"],
        "bool col": [True, False, False, True, True],
        "categorical with nan": pd.Series([0, 1, 0, 0, 3], dtype='category'),
        "object with nan": ["b", "b", "b", "c", "b"],
        "bool col with nan": [True, True, False, True, True]
    })
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TypedImputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_drop_all_columns():
    X = pd.DataFrame({
        "all nan cat": pd.Series([np.nan, np.nan, np.nan, np.nan, np.nan], dtype='category'),
        "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan]
    })
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TypedImputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = X.drop(["all nan cat", "all nan"], axis=1)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TypedImputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_typed_imputer_numpy_input():
    X = np.array([[1, 2, 2, 0],
                  [np.nan, 0, 0, 0],
                  [1, np.nan, np.nan, np.nan]])
    y = pd.Series([0, 0, 1])
    imputer = TypedImputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(np.array([[1, 2, 2, 0],
                                      [1, 0, 0, 0],
                                      [1, 1, 1, 0]]))
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TypedImputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)
