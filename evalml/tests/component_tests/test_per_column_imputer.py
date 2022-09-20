import warnings

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import (
    Boolean,
    Categorical,
    Double,
    Integer,
    NaturalLanguage,
)

from evalml.pipelines.components import PerColumnImputer
from evalml.utils.woodwork_utils import infer_feature_types


@pytest.fixture
def non_numeric_df():
    X = pd.DataFrame(
        [
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["a", "a", "a", "a"],
            [np.nan, np.nan, np.nan, np.nan],
        ],
    )
    X.columns = ["A", "B", "C", "D"]
    return X


def test_invalid_parameters():
    with pytest.raises(ValueError):
        strategies = ("impute_strategy", "mean")
        PerColumnImputer(impute_strategies=strategies)

    with pytest.raises(ValueError):
        strategies = ["mean"]
        PerColumnImputer(impute_strategies=strategies)


def test_all_strategies():
    X = pd.DataFrame(
        {
            "A": pd.Series([2, 4, 6, np.nan]),
            "B": pd.Series([4, 6, 4, np.nan]),
            "C": pd.Series([6, 8, 8, np.nan]),
            "D": pd.Series(["a", "a", "b", np.nan]),
        },
    )
    X.ww.init(logical_types={"D": "categorical"})

    X_expected = pd.DataFrame(
        {
            "A": pd.Series([2, 4, 6, 4]),
            "B": pd.Series([4, 6, 4, 4]),
            "C": pd.Series([6, 8, 8, 100]),
            "D": pd.Series(["a", "a", "b", "a"], dtype="category"),
        },
    )

    strategies = {
        "A": {"impute_strategy": "mean"},
        "B": {"impute_strategy": "median"},
        "C": {"impute_strategy": "constant", "fill_value": 100},
        "D": {"impute_strategy": "most_frequent"},
    }

    transformer = PerColumnImputer(impute_strategies=strategies)
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t, check_dtype=False)


def test_fit_transform():
    X = pd.DataFrame([[2], [4], [6], [np.nan]])

    X_expected = pd.DataFrame([[2], [4], [6], [4]])

    X.columns = ["A"]
    X_expected.columns = ["A"]
    strategies = {"A": {"impute_strategy": "median"}}

    transformer = PerColumnImputer(impute_strategies=strategies)
    transformer.fit(X)
    X_t = transformer.transform(X)

    transformer = PerColumnImputer(impute_strategies=strategies)
    X_fit_transform = transformer.fit_transform(X)

    assert_frame_equal(X_t, X_fit_transform)


def test_non_numeric_errors(non_numeric_df):
    # test col with all strings
    X = non_numeric_df
    X.ww.init(
        logical_types={
            "A": "categorical",
            "B": "categorical",
            "C": "categorical",
            "D": "categorical",
        },
    )
    # mean with all strings
    strategies = {"A": {"impute_strategy": "mean"}}
    with pytest.raises(
        ValueError,
        match="Cannot use mean strategy with non-numeric data",
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit_transform(X)
    with pytest.raises(
        ValueError,
        match="Cannot use mean strategy with non-numeric data",
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit(X)

    # median with all strings
    strategies = {"B": {"impute_strategy": "median"}}
    with pytest.raises(
        ValueError,
        match="Cannot use median strategy with non-numeric data",
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit_transform(X)
    with pytest.raises(
        ValueError,
        match="Cannot use median strategy with non-numeric data",
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit(X)


def test_non_numeric_valid(non_numeric_df):
    X = non_numeric_df.copy()
    X.ww.init(
        logical_types={
            "A": "categorical",
            "B": "categorical",
            "C": "categorical",
            "D": "categorical",
        },
    )
    # most frequent with all strings
    strategies = {
        "A": {"impute_strategy": "most_frequent"},
        "B": {"impute_strategy": "most_frequent"},
        "C": {"impute_strategy": "most_frequent"},
        "D": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)

    X_expected = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "B": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "C": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "D": pd.Series(["a", "b", "a", "a"], dtype="category"),
        },
    )

    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t)

    X = non_numeric_df.copy()
    # constant with all strings
    strategies = {
        "B": {"impute_strategy": "most_frequent"},
        "C": {"impute_strategy": "most_frequent"},
        "D": {"impute_strategy": "constant", "fill_value": 100},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)

    X.ww.init(
        logical_types={
            "A": "categorical",
            "B": "categorical",
            "C": "categorical",
            "D": "categorical",
        },
    )
    X_expected = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", np.nan], dtype="category"),
            "B": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "C": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "D": pd.Series(["a", "b", "a", 100], dtype="category"),
        },
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t)


def test_datetime_does_not_error(fraud_100):
    X, y = fraud_100
    pci = PerColumnImputer(
        impute_strategies={"country": {"impute_strategy": "most_frequent"}},
    )
    pci.fit(X, y)

    assert pci._is_fitted


def test_fit_transform_drop_all_nan_columns(imputer_test_data):
    X = imputer_test_data.ww[["all nan", "int col", "int with nan"]]
    strategies = {
        "all nan": {"impute_strategy": "most_frequent"},
        "int with nan": {"impute_strategy": "most_frequent"},
        "int col": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    X_expected_arr = pd.DataFrame(
        {
            "int col": [0, 1, 2, 0, 3] * 4,
            "int with nan": [0, 1, 0, 0, 1] * 4,
        },
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
    assert "all nan" in X.columns


def test_transform_drop_all_nan_columns(imputer_test_data):
    X = imputer_test_data.ww[["all nan", "int col", "int with nan"]]
    strategies = {
        "all nan": {"impute_strategy": "most_frequent"},
        "int with nan": {"impute_strategy": "most_frequent"},
        "int col": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    transformer.fit(X)
    X_expected_arr = pd.DataFrame(
        {
            "int col": [0, 1, 2, 0, 3] * 4,
            "int with nan": [0, 1, 0, 0, 1] * 4,
        },
    )
    X_t = transformer.transform(X)

    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
    assert "all nan" in X.columns


def test_transform_drop_all_nan_columns_empty():
    X = pd.DataFrame([[np.nan, np.nan, np.nan]])
    strategies = {i: {"impute_strategy": "most_frequent"} for i in range(3)}
    X.ww.init(logical_types={0: "Double", 1: "Double", 2: "Double"})
    transformer = PerColumnImputer(impute_strategies=strategies)
    assert transformer.fit_transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))

    strategies = {i: {"impute_strategy": "most_frequent"} for i in range(3)}
    transformer = PerColumnImputer(impute_strategies=strategies)
    transformer.fit(X)
    assert transformer.transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(pd.Series([1, 2, 3], dtype="int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 3.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"], dtype="category")),
        pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
        pd.DataFrame(
            pd.Series(
                ["this will be a natural language column because length", "yay", "hay"],
                dtype="string",
            ),
        ),
    ],
)
@pytest.mark.parametrize("has_nan", [True, False])
def test_per_column_imputer_woodwork_custom_overrides_returned_by_components(
    X_df,
    has_nan,
):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean]
    for logical_type in override_types:
        # Column with Nans to boolean used to fail. Now it doesn't
        if has_nan and logical_type in [Boolean, NaturalLanguage]:
            continue
        try:
            X = X_df.copy()
            if has_nan:
                X.iloc[len(X_df) - 1, 0] = np.nan
            X.ww.init(logical_types={0: logical_type})
        except ww.exceptions.TypeConversionError:
            continue

        imputer = PerColumnImputer()
        imputer.fit(X, y)
        transformed = imputer.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            0: logical_type,
        }


def test_per_column_imputer_column_subset():
    X = pd.DataFrame(
        {
            "all_nan_not_included": [np.nan, np.nan, np.nan],
            "all_nan_included": [np.nan, np.nan, np.nan],
            "column_with_nan_not_included": [np.nan, 1, 0],
            "column_with_nan_included": [0, 1, np.nan],
        },
    )
    X.ww.init(
        logical_types={
            "column_with_nan_not_included": "IntegerNullable",
            "column_with_nan_included": "IntegerNullable",
        },
    )
    strategies = {
        "all_nan_included": {"impute_strategy": "most_frequent"},
        "column_with_nan_included": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    X_expected = pd.DataFrame(
        {
            "all_nan_not_included": [np.nan, np.nan, np.nan],
            "column_with_nan_not_included": [np.nan, 1, 0],
            "column_with_nan_included": [0, 1, 0],
        },
    )
    X_expected.ww.init(
        logical_types={
            "all_nan_not_included": "Double",
            "column_with_nan_not_included": "IntegerNullable",
            "column_with_nan_included": "IntegerNullable",
        },
    )
    X.ww.init(
        logical_types={"all_nan_included": "Double", "all_nan_not_included": "Double"},
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t)
    assert_frame_equal(
        X,
        pd.DataFrame(
            {
                "all_nan_not_included": [np.nan, np.nan, np.nan],
                "all_nan_included": [np.nan, np.nan, np.nan],
                "column_with_nan_not_included": [pd.NA, 1, 0],
                "column_with_nan_included": [0, 1, 0],
            },
        ).astype({"column_with_nan_not_included": "Int64"}),
    )


def test_per_column_imputer_impute_strategies_is_None():
    X = pd.DataFrame(
        {
            "all_nan_not_included": [np.nan, np.nan, np.nan],
            "all_nan_included": [np.nan, np.nan, np.nan],
            "column_with_nan_not_included": [np.nan, 1, 0],
            "column_with_nan_included": [0, 1, np.nan],
        },
    )
    X_expected = infer_feature_types(X)
    transformer = PerColumnImputer(impute_strategies=None)

    X_t = None
    with warnings.catch_warnings(record=True) as w:
        X_t = transformer.fit_transform(X)
    assert len(w) == 1
    assert "No columns to impute. Please check `impute_strategies` parameter." in str(
        w[-1].message,
    )
    assert_frame_equal(X_expected, X_t)
