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


@pytest.fixture
def non_numeric_df():
    X = pd.DataFrame(
        [
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["a", "a", "a", "a"],
            [np.nan, np.nan, np.nan, np.nan],
        ]
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
        }
    )
    X.ww.init(logical_types={"D": "categorical"})

    X_expected = pd.DataFrame(
        {
            "A": pd.Series([2, 4, 6, 4]),
            "B": pd.Series([4, 6, 4, 4]),
            "C": pd.Series([6, 8, 8, 100]),
            "D": pd.Series(["a", "a", "b", "a"], dtype="category"),
        }
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
        }
    )
    # mean with all strings
    strategies = {"A": {"impute_strategy": "mean"}}
    with pytest.raises(
        ValueError, match="Cannot use mean strategy with non-numeric data"
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit_transform(X)
    with pytest.raises(
        ValueError, match="Cannot use mean strategy with non-numeric data"
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit(X)

    # median with all strings
    strategies = {"B": {"impute_strategy": "median"}}
    with pytest.raises(
        ValueError, match="Cannot use median strategy with non-numeric data"
    ):
        transformer = PerColumnImputer(impute_strategies=strategies)
        transformer.fit_transform(X)
    with pytest.raises(
        ValueError, match="Cannot use median strategy with non-numeric data"
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
        }
    )
    # most frequent with all strings
    strategies = {"C": {"impute_strategy": "most_frequent"}}
    transformer = PerColumnImputer(impute_strategies=strategies)

    X_expected = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "B": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "C": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "D": pd.Series(["a", "b", "a", "a"], dtype="category"),
        }
    )

    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t)

    X = non_numeric_df.copy()
    # constant with all strings
    strategies = {"D": {"impute_strategy": "constant", "fill_value": 100}}
    transformer = PerColumnImputer(impute_strategies=strategies)

    X.ww.init(
        logical_types={
            "A": "categorical",
            "B": "categorical",
            "C": "categorical",
            "D": "categorical",
        }
    )
    X_expected = pd.DataFrame(
        [
            ["a", "a", "a", "a"],
            ["b", "b", "b", "b"],
            ["a", "a", "a", "a"],
            ["a", "a", "a", 100],
        ]
    )
    X_expected.columns = ["A", "B", "C", "D"]
    X_expected = pd.DataFrame(
        {
            "A": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "B": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "C": pd.Series(["a", "b", "a", "a"], dtype="category"),
            "D": pd.Series(["a", "b", "a", 100], dtype="category"),
        }
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t)


def test_fit_transform_drop_all_nan_columns():
    X = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "some_nan": [np.nan, 1, 0],
            "another_col": [0, 1, 2],
        }
    )
    strategies = {
        "all_nan": {"impute_strategy": "most_frequent"},
        "some_nan": {"impute_strategy": "most_frequent"},
        "another_col": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
    assert_frame_equal(
        X,
        pd.DataFrame(
            {
                "all_nan": [np.nan, np.nan, np.nan],
                "some_nan": [np.nan, 1, 0],
                "another_col": [0, 1, 2],
            }
        ),
    )


def test_transform_drop_all_nan_columns():
    X = pd.DataFrame(
        {
            "all_nan": [np.nan, np.nan, np.nan],
            "some_nan": [np.nan, 1, 0],
            "another_col": [0, 1, 2],
        }
    )
    strategies = {
        "all_nan": {"impute_strategy": "most_frequent"},
        "some_nan": {"impute_strategy": "most_frequent"},
        "another_col": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    transformer.fit(X)
    X_expected_arr = pd.DataFrame({"some_nan": [0, 1, 0], "another_col": [0, 1, 2]})
    X_t = transformer.transform(X)

    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)
    assert_frame_equal(
        X,
        pd.DataFrame(
            {
                "all_nan": [np.nan, np.nan, np.nan],
                "some_nan": [np.nan, 1, 0],
                "another_col": [0, 1, 2],
            }
        ),
    )


def test_transform_drop_all_nan_columns_empty():
    X = pd.DataFrame([[np.nan, np.nan, np.nan]])
    strategies = {
        "0": {"impute_strategy": "most_frequent"},
    }
    transformer = PerColumnImputer(impute_strategies=strategies)
    assert transformer.fit_transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))

    strategies = {"0": {"impute_strategy": "most_frequent"}}
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
            )
        ),
    ],
)
@pytest.mark.parametrize("has_nan", [True, False])
def test_per_column_imputer_woodwork_custom_overrides_returned_by_components(
    X_df, has_nan
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
            0: logical_type
        }
