from unittest.mock import patch

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

from evalml.pipelines.components import Imputer


@pytest.fixture
def imputer_test_data():
    return pd.DataFrame(
        {
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4, dtype="category"
            ),
            "int col": [0, 1, 2, 0, 3] * 4,
            "object col": ["b", "b", "a", "c", "d"] * 4,
            "float col": [0.0, 1.0, 0.0, -2.0, 5.0] * 4,
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                [np.nan, "1", "0", "0", "3"] * 4, dtype="category"
            ),
            "int with nan": [np.nan, 1, 0, 0, 1] * 4,
            "float with nan": [0.0, 1.0, np.nan, -1.0, 0.0] * 4,
            "object with nan": ["b", "b", np.nan, "c", np.nan] * 4,
            "bool col with nan": pd.Series(
                [True, np.nan, False, np.nan, True] * 4, dtype="category"
            ),
            "all nan": [np.nan, np.nan, np.nan, np.nan, np.nan] * 4,
            "all nan cat": pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan] * 4, dtype="category"
            ),
        }
    )


def test_invalid_strategy_parameters():
    with pytest.raises(ValueError, match="Valid impute strategies are"):
        Imputer(numeric_impute_strategy="not a valid strategy")
    with pytest.raises(ValueError, match="Valid categorical impute strategies are"):
        Imputer(categorical_impute_strategy="mean")


def test_imputer_default_parameters():
    imputer = Imputer()
    expected_parameters = {
        "categorical_impute_strategy": "most_frequent",
        "numeric_impute_strategy": "mean",
        "categorical_fill_value": None,
        "numeric_fill_value": None,
    }
    assert imputer.parameters == expected_parameters


@pytest.mark.parametrize("categorical_impute_strategy", ["most_frequent", "constant"])
@pytest.mark.parametrize(
    "numeric_impute_strategy", ["mean", "median", "most_frequent", "constant"]
)
def test_imputer_init(categorical_impute_strategy, numeric_impute_strategy):

    imputer = Imputer(
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_fill_value="str_fill_value",
        numeric_fill_value=-1,
    )
    expected_parameters = {
        "categorical_impute_strategy": categorical_impute_strategy,
        "numeric_impute_strategy": numeric_impute_strategy,
        "categorical_fill_value": "str_fill_value",
        "numeric_fill_value": -1,
    }
    expected_hyperparameters = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent"],
    }
    assert imputer.name == "Imputer"
    assert imputer.parameters == expected_parameters
    assert imputer.hyperparameter_ranges == expected_hyperparameters


def test_numeric_only_input(imputer_test_data):
    X = imputer_test_data[
        ["int col", "float col", "int with nan", "float with nan", "all nan"]
    ]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = Imputer(numeric_impute_strategy="median")
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "int col": [0, 1, 2, 0, 3] * 4,
            "float col": [0.0, 1.0, 0.0, -2.0, 5.0] * 4,
            "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0] * 4,
            "float with nan": [0.0, 1.0, 0, -1.0, 0.0] * 4,
        }
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_only_input(imputer_test_data):
    X = imputer_test_data[
        [
            "categorical col",
            "object col",
            "bool col",
            "categorical with nan",
            "object with nan",
            "bool col with nan",
            "all nan cat",
        ]
    ]
    y = pd.Series([0, 0, 1, 0, 1] * 4)

    expected = pd.DataFrame(
        {
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4, dtype="category"
            ),
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                ["0", "1", "0", "0", "3"] * 4, dtype="category"
            ),
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "b"] * 4, dtype="category"
            ),
            "bool col with nan": pd.Series(
                [True, True, False, True, True] * 4, dtype="category"
            ),
        }
    )

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)

    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_and_numeric_input(imputer_test_data):
    X = imputer_test_data
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4, dtype="category"
            ),
            "int col": [0, 1, 2, 0, 3] * 4,
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "float col": [0.0, 1.0, 0.0, -2.0, 5.0] * 4,
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                ["0", "1", "0", "0", "3"] * 4, dtype="category"
            ),
            "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0] * 4,
            "float with nan": [0.0, 1.0, 0, -1.0, 0.0] * 4,
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "b"] * 4, dtype="category"
            ),
            "bool col with nan": pd.Series(
                [True, True, False, True, True] * 4, dtype="category"
            ),
        }
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_drop_all_columns(imputer_test_data):
    X = imputer_test_data[["all nan cat", "all nan"]]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    X.ww.init()
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = X.drop(["all nan cat", "all nan"], axis=1)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_typed_imputer_numpy_input():
    X = np.array([[1, 2, 2, 0], [np.nan, 0, 0, 0], [1, np.nan, np.nan, np.nan]])
    y = pd.Series([0, 0, 1])
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(np.array([[1, 2, 2, 0], [1, 0, 0, 0], [1, 1, 1, 0]]))
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_datetime_input():
    X = pd.DataFrame(
        {
            "dates": ["20190902", "20200519", "20190607", np.nan],
            "more dates": ["20190902", "20201010", "20190921", np.nan],
        }
    )
    X["dates"] = pd.to_datetime(X["dates"], format="%Y%m%d")
    X["more dates"] = pd.to_datetime(X["more dates"], format="%Y%m%d")
    y = pd.Series()

    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(transformed, X, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, X, check_dtype=False)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_imputer_empty_data(data_type, make_data_type):
    X = pd.DataFrame()
    y = pd.Series()
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    expected = pd.DataFrame(index=pd.Index([]), columns=pd.Index([]))
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_does_not_reset_index():
    X = pd.DataFrame(
        {
            "input_val": np.arange(10),
            "target": np.arange(10),
            "input_cat": ["a"] * 7 + ["b"] * 3,
        }
    )
    X.loc[5, "input_val"] = np.nan
    X.loc[5, "input_cat"] = np.nan
    assert X.index.tolist() == list(range(10))
    X.ww.init(logical_types={"input_cat": "categorical"})

    X.drop(0, inplace=True)
    y = X.ww.pop("target")

    imputer = Imputer()
    imputer.fit(X, y=y)
    transformed = imputer.transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame(
            {
                "input_val": [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
                "input_cat": pd.Categorical(["a"] * 6 + ["b"] * 3),
            },
            index=list(range(1, 10)),
        ),
    )


def test_imputer_fill_value(imputer_test_data):
    X = imputer_test_data[
        [
            "int with nan",
            "categorical with nan",
            "float with nan",
            "object with nan",
            "bool col with nan",
        ]
    ]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = Imputer(
        categorical_impute_strategy="constant",
        numeric_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
    )
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "int with nan": [-1, 1, 0, 0, 1] * 4,
            "categorical with nan": pd.Series(
                ["fill", "1", "0", "0", "3"] * 4, dtype="category"
            ),
            "float with nan": [0.0, 1.0, -1, -1.0, 0.0] * 4,
            "object with nan": pd.Series(
                ["b", "b", "fill", "c", "fill"] * 4, dtype="category"
            ),
            "bool col with nan": pd.Series(
                [True, "fill", False, "fill", True] * 4, dtype="category"
            ),
        }
    )
    assert_frame_equal(expected, transformed, check_dtype=False)

    imputer = Imputer(
        categorical_impute_strategy="constant",
        numeric_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
    )
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(expected, transformed, check_dtype=False)


def test_imputer_no_nans(imputer_test_data):
    X = imputer_test_data[["categorical col", "object col", "bool col"]]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = Imputer(
        categorical_impute_strategy="constant",
        numeric_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
    )
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4, dtype="category"
            ),
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "bool col": [True, False, False, True, True] * 4,
        }
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(
        categorical_impute_strategy="constant",
        numeric_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
    )
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_imputer_with_none():
    X = pd.DataFrame(
        {
            "int with None": [1, 0, 5, None] * 4,
            "float with None": [0.1, 0.0, 0.5, None] * 4,
            "category with None": pd.Series(
                ["b", "a", "a", None] * 4, dtype="category"
            ),
            "boolean with None": pd.Series([True, None, False, True] * 4),
            "object with None": ["b", "a", "a", None] * 4,
            "all None": [None, None, None, None] * 4,
        }
    )
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = Imputer()
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "int with None": [1, 0, 5, 2] * 4,
            "float with None": [0.1, 0.0, 0.5, 0.2] * 4,
            "category with None": pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
            "boolean with None": pd.Series(
                [True, True, False, True] * 4, dtype="category"
            ),
            "object with None": pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
        }
    )
    assert_frame_equal(expected, transformed, check_dtype=False)

    imputer = Imputer()
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(expected, transformed, check_dtype=False)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_all_bool_return_original(data_type, make_data_type):
    X = make_data_type(
        data_type, pd.DataFrame([True, True, False, True, True], dtype=bool)
    )
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype=bool)
    y = make_data_type(data_type, pd.Series([1, 0, 0, 1, 0]))

    imputer = Imputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_bool_dtype_object(data_type, make_data_type):
    X = pd.DataFrame([True, np.nan, False, np.nan, True] * 4)
    y = pd.Series([1, 0, 0, 1, 0] * 4)
    X_expected_arr = pd.DataFrame([True, True, False, True, True] * 4, dtype="category")
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    imputer = Imputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_multitype_with_one_bool(data_type, make_data_type):
    X_multi = pd.DataFrame(
        {
            "bool with nan": pd.Series([True, np.nan, False, np.nan, False] * 4),
            "bool no nan": pd.Series(
                [False, False, False, False, True] * 4, dtype=bool
            ),
        }
    )
    y = pd.Series([1, 0, 0, 1, 0] * 4)
    X_multi_expected_arr = pd.DataFrame(
        {
            "bool with nan": pd.Series(
                [True, False, False, False, False] * 4, dtype="category"
            ),
            "bool no nan": pd.Series(
                [False, False, False, False, True] * 4, dtype=bool
            ),
        }
    )

    X_multi = make_data_type(data_type, X_multi)
    y = make_data_type(data_type, y)

    imputer = Imputer()
    imputer.fit(X_multi, y)
    X_multi_t = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t)


def test_imputer_int_preserved():
    X = pd.DataFrame(pd.Series([1, 2, 11, np.nan]))
    imputer = Imputer(numeric_impute_strategy="mean")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed, pd.DataFrame(pd.Series([1, 2, 11, 14 / 3]))
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {0: Double}

    X = pd.DataFrame(pd.Series([1, 2, 3, np.nan]))
    imputer = Imputer(numeric_impute_strategy="mean")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed, pd.DataFrame(pd.Series([1, 2, 3, 2])), check_dtype=False
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {0: Double}

    X = pd.DataFrame(pd.Series([1, 2, 3, 4], dtype="int"))
    imputer = Imputer(numeric_impute_strategy="mean")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed, pd.DataFrame(pd.Series([1, 2, 3, 4])), check_dtype=False
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {0: Integer}


def test_imputer_bool_preserved():
    X = pd.DataFrame(pd.Series([True, False, True, np.nan] * 4))
    imputer = Imputer(categorical_impute_strategy="most_frequent")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame(pd.Series([True, False, True, True] * 4, dtype="category")),
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
        0: Categorical
    }

    X = pd.DataFrame(pd.Series([True, False, True, False] * 4))
    imputer = Imputer(categorical_impute_strategy="most_frequent")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame(pd.Series([True, False, True, False] * 4)),
        check_dtype=False,
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {0: Boolean}


def test_imputer_does_not_erase_ww_info():
    df_train = pd.DataFrame({"a": [1, 2, 3, 2], "b": ["a", "b", "b", "c"]})
    df_holdout = pd.DataFrame({"a": [2], "b": [None]})
    df_train.ww.init(logical_types={"a": "Double", "b": "Categorical"})
    df_holdout.ww.init(logical_types={"a": "Double", "b": "Categorical"})

    imputer = Imputer()
    imputer.fit(df_train, None)
    # Would error out if ww got erased because `b` would be inferred as Unknown, then Double.
    imputer.transform(df_holdout, None)

    with patch("evalml.pipelines.components.SimpleImputer.transform") as mock_transform:
        mock_transform.side_effect = [df_holdout[["a"]], df_train[["b"]].iloc[0]]
        imputer.transform(df_holdout, None)
    mock_transform.call_args[0][0].ww.schema == df_holdout.ww[["b"]].ww.schema


@pytest.mark.parametrize(
    "X_df",
    [
        pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
        pd.DataFrame(pd.Series([1.0, 2.0, 4.0], dtype="float")),
        pd.DataFrame(pd.Series(["a", "b", "a"], dtype="category")),
        pd.DataFrame(pd.Series([True, False, True], dtype=bool)),
        pd.DataFrame(
            pd.Series(
                ["this will be a natural language column because length", "yay", "hay"],
                dtype="string",
            )
        ),
    ],
)
@pytest.mark.parametrize("has_nan", [True, False])
@pytest.mark.parametrize("numeric_impute_strategy", ["mean", "median", "most_frequent"])
def test_imputer_woodwork_custom_overrides_returned_by_components(
    X_df, has_nan, numeric_impute_strategy
):
    y = pd.Series([1, 2, 1])
    override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean]
    for logical_type in override_types:
        # Column with Nans to boolean used to fail. Now it doesn't but it should.
        if has_nan and logical_type == Boolean:
            continue
        try:
            X = X_df.copy()
            if has_nan:
                X.iloc[len(X_df) - 1, 0] = np.nan
            X.ww.init(logical_types={0: logical_type})
        except ww.exceptions.TypeConversionError:
            continue

        imputer = Imputer(numeric_impute_strategy=numeric_impute_strategy)
        imputer.fit(X, y)
        transformed = imputer.transform(X, y)
        assert isinstance(transformed, pd.DataFrame)
        if numeric_impute_strategy == "most_frequent":
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: logical_type
            }
        elif logical_type in [Categorical, NaturalLanguage] or not has_nan:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: logical_type
            }
        else:
            assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
                0: Double
            }
