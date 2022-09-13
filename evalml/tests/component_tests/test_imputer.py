from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Categorical,
    Double,
    NaturalLanguage,
)

from evalml.pipelines.components import Imputer
from evalml.pipelines.components.transformers.imputers import KNNImputer, SimpleImputer


def test_invalid_strategy_parameters():
    with pytest.raises(ValueError, match="Valid numeric imputation strategies are"):
        Imputer(numeric_impute_strategy="not a valid strategy")
    with pytest.raises(ValueError, match="Valid categorical imputation strategies are"):
        Imputer(categorical_impute_strategy="mean")
    with pytest.raises(ValueError, match="Valid boolean imputation strategies are"):
        Imputer(boolean_impute_strategy="another invalid strategy")


def test_imputer_default_parameters():
    imputer = Imputer()
    expected_parameters = {
        "categorical_impute_strategy": "most_frequent",
        "numeric_impute_strategy": "mean",
        "boolean_impute_strategy": "most_frequent",
        "categorical_fill_value": None,
        "numeric_fill_value": None,
        "boolean_fill_value": None,
    }
    assert imputer.parameters == expected_parameters


@pytest.mark.parametrize("categorical_impute_strategy", ["most_frequent", "constant"])
@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["mean", "median", "most_frequent", "constant"],
)
@pytest.mark.parametrize("boolean_impute_strategy", ["most_frequent", "constant"])
def test_imputer_init(
    categorical_impute_strategy,
    numeric_impute_strategy,
    boolean_impute_strategy,
):

    imputer = Imputer(
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_impute_strategy=numeric_impute_strategy,
        boolean_impute_strategy=boolean_impute_strategy,
        categorical_fill_value="str_fill_value",
        numeric_fill_value=-1,
        boolean_fill_value=True,
    )
    expected_parameters = {
        "categorical_impute_strategy": categorical_impute_strategy,
        "numeric_impute_strategy": numeric_impute_strategy,
        "boolean_impute_strategy": boolean_impute_strategy,
        "categorical_fill_value": "str_fill_value",
        "numeric_fill_value": -1,
        "boolean_fill_value": True,
    }
    expected_hyperparameters = {
        "categorical_impute_strategy": ["most_frequent"],
        "numeric_impute_strategy": ["mean", "median", "most_frequent", "knn"],
        "boolean_impute_strategy": ["most_frequent", "knn"],
    }
    assert imputer.name == "Imputer"
    assert imputer.parameters == expected_parameters
    assert imputer.hyperparameter_ranges == expected_hyperparameters


@pytest.mark.parametrize("categorical_impute_strategy", ["most_frequent", "constant"])
def test_knn_as_input(categorical_impute_strategy):
    imputer = Imputer(
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_impute_strategy="knn",
        boolean_impute_strategy="knn",
    )
    assert isinstance(imputer._categorical_imputer, SimpleImputer)
    assert isinstance(imputer._numeric_imputer, KNNImputer)
    assert isinstance(imputer._boolean_imputer, KNNImputer)

    expected_numeric_parameters = {
        "number_neighbors": 3,
    }
    expected_boolean_parameters = {
        "number_neighbors": 1,
    }

    assert imputer._numeric_imputer.parameters == expected_numeric_parameters
    assert imputer._boolean_imputer.parameters == expected_boolean_parameters


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
            "float col": [0.1, 1.0, 0.0, -2.0, 5.0] * 4,
            "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0] * 4,
            "float with nan": [0.3, 1.0, 0.15, -1.0, 0.0] * 4,
        },
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = Imputer(numeric_impute_strategy="median")
    transformed = imputer.fit_transform(X, y)
    expected["float with nan"] = [0.3, 1.0, 0.15, -1.0, 0.0] * 4
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
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                ["0", "1", "0", "0", "3"] * 4,
                dtype="category",
            ),
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "b"] * 4,
                dtype="category",
            ),
            "bool col with nan": pd.Series(
                [True, True, False, True, True] * 4,
            ),
        },
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
            "dates": pd.date_range("01-01-2022", periods=20),
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "int col": [0, 1, 2, 0, 3] * 4,
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "float col": [0.1, 1.0, 0.0, -2.0, 5.0] * 4,
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                ["0", "1", "0", "0", "3"] * 4,
                dtype="category",
            ),
            "int with nan": [0.5, 1.0, 0.0, 0.0, 1.0] * 4,
            "float with nan": [0.3, 1.0, 0.075, -1.0, 0.0] * 4,
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "b"] * 4,
                dtype="category",
            ),
            "bool col with nan": pd.Series(
                [True, True, False, True, True] * 4,
            ),
            "natural language col": pd.Series(
                ["cats are really great", "don't", "believe", "me?", "well..."] * 4,
                dtype="string",
            ),
        },
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
        },
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
        },
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
        boolean_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
        boolean_fill_value=True,
    )
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "int with nan": [-1, 1, 0, 0, 1] * 4,
            "categorical with nan": pd.Series(
                ["fill", "1", "0", "0", "3"] * 4,
                dtype="category",
            ),
            "float with nan": [0.3, 1.0, -1, -1.0, 0.0] * 4,
            "object with nan": pd.Series(
                ["b", "b", "fill", "c", "fill"] * 4,
                dtype="category",
            ),
            "bool col with nan": pd.Series(
                [True, True, False, True, True] * 4,
            ),
        },
    )
    assert_frame_equal(expected, transformed, check_dtype=False)

    imputer = Imputer(
        categorical_impute_strategy="constant",
        numeric_impute_strategy="constant",
        boolean_impute_strategy="constant",
        categorical_fill_value="fill",
        numeric_fill_value=-1,
        boolean_fill_value=True,
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
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "bool col": [True, False, False, True, True] * 4,
        },
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


X = pd.DataFrame(
    {
        "int with None": [1, 0, 5, 5, None],
        "float with None": [0.1, 0.0, 0.5, 0.5, None],
        "category with None": pd.Series(["c", "a", "a", "a", None], dtype="category"),
        "boolean with None": pd.Series([True, None, False, True, True]),
        "object with None": ["b", "a", "a", "a", None],
        "all None": [None, None, None, None, None],
    },
)

expected = pd.DataFrame(
    {
        "int with None": pd.Series([1, 0, 5, 5, 5], dtype="Int64"),
        "float with None": [0.1, 0.0, 0.5, 0.5, 0.5],
        "category with None": pd.Series(["c", "a", "a", "a", "a"], dtype="category"),
        "boolean with None": pd.Series(
            [True, True, False, True, True],
            dtype="boolean",
        ),
        "object with None": pd.Series(["b", "a", "a", "a", "a"], dtype="category"),
    },
)
ltypes = {
    "int with None": "IntegerNullable",
    "float with None": "Double",
    "category with None": "categorical",
    "boolean with None": "BooleanNullable",
    "object with None": "categorical",
    "all None": "categorical",
}
columns_dict = {
    "integers_only": ["int with None"],
    "floats_only": ["float with None"],
    "numerics_only": ["int with None", "float with None"],
    "categoricals_only": ["category with None", "object with None"],
    "booleans_only": ["boolean with None"],
    "categorical_and_booleans": [
        "category with None",
        "boolean with None",
        "object with None",
    ],
    "all": [
        "int with None",
        "float with None",
        "category with None",
        "boolean with None",
        "object with None",
    ],
}


@pytest.mark.parametrize(
    "dtypes",
    [
        "integers_only",
        "floats_only",
        "numerics_only",
        "booleans_only",
        "categoricals_only",
        "categorical_and_booleans",
        "all",
    ],
)
@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["most_frequent", "mean", "median", "constant"],
)
@pytest.mark.parametrize("categorical_impute_strategy", ["most_frequent", "constant"])
@pytest.mark.parametrize("boolean_impute_strategy", ["most_frequent", "constant"])
def test_imputer_with_none_separated(
    dtypes,
    numeric_impute_strategy,
    categorical_impute_strategy,
    boolean_impute_strategy,
):
    """Test for correctness for the different numeric, categorical and boolean dtypes using dataframes that contain
    either just the tested imputed dtypes or combinations of dtypes."""

    test_ltypes = dict((k, ltypes[k]) for k in columns_dict[dtypes])
    X_test = X[columns_dict[dtypes]]
    X_test.ww.init(logical_types=test_ltypes)
    y = pd.Series([0, 0, 1, 0, 1])
    numeric_fill_value = 0
    categorical_fill_value = "filler"
    boolean_fill_value = True
    imputer = Imputer(
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_fill_value=numeric_fill_value,
        categorical_fill_value=categorical_fill_value,
        boolean_impute_strategy=boolean_impute_strategy,
        boolean_fill_value=boolean_fill_value,
    )
    imputer.fit(X_test, y)
    transformed = imputer.transform(X_test, y)

    # Build the expected dataframe
    expected_columns = columns_dict[dtypes]
    expected_df = deepcopy(expected[expected_columns])
    if numeric_impute_strategy in ["mean", "median"]:
        for col in set(columns_dict["numerics_only"]).intersection(set(X_test.columns)):
            expected_df = expected_df.astype({col: float})
            if numeric_impute_strategy == "mean":
                expected_df[col].iloc[-1:] = X_test[col].mean()
            elif numeric_impute_strategy == "median":
                expected_df[col].iloc[-1:] = X_test[col].median()
    elif numeric_impute_strategy == "constant":
        for col in set(columns_dict["numerics_only"]).intersection(set(X_test.columns)):
            expected_df[col].iloc[-1:] = numeric_fill_value
    if categorical_impute_strategy == "constant":
        for col in set(columns_dict["categoricals_only"]).intersection(
            set(X_test.columns),
        ):
            expected_df[col].cat.add_categories(categorical_fill_value, inplace=True)
            expected_df[col].iloc[-1:] = categorical_fill_value
    if boolean_impute_strategy == "constant":
        for col in set(columns_dict["booleans_only"]).intersection(set(X_test.columns)):
            expected_df[col].iloc[-1:] = boolean_fill_value
    assert_frame_equal(expected_df, transformed, check_dtype=False)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_all_bool_return_original(data_type, make_data_type):
    X = make_data_type(
        data_type,
        pd.DataFrame([True, True, False, True, True], dtype=bool),
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
    X.ww.init(logical_types={0: BooleanNullable})
    y = pd.Series([1, 0, 0, 1, 0] * 4)
    X_expected_arr = pd.DataFrame([True, True, False, True, True] * 4, dtype="bool")
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
                [False, False, False, False, True] * 4,
                dtype=bool,
            ),
        },
    )
    X_multi.ww.init(
        logical_types={"bool with nan": BooleanNullable, "bool no nan": Boolean},
    )
    y = pd.Series([1, 0, 0, 1, 0] * 4)
    X_multi_expected_arr = pd.DataFrame(
        {
            "bool with nan": pd.Series(
                [True, False, False, False, False] * 4,
                dtype="bool",
            ),
            "bool no nan": pd.Series(
                [False, False, False, False, True] * 4,
            ),
        },
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
        transformed,
        pd.DataFrame(pd.Series([1, 2, 11, 14 / 3])),
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
        0: Double,
    }

    X = pd.DataFrame(pd.Series([1, 2, 3, np.nan]))
    imputer = Imputer(numeric_impute_strategy="mean")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame(pd.Series([1, 2, 3, 2])),
        check_dtype=False,
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
        0: Double,
    }

    X = pd.DataFrame(pd.Series([1, 2, 3, 4], dtype="int"))
    imputer = Imputer(numeric_impute_strategy="mean")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        pd.DataFrame(pd.Series([1, 2, 3, 4])),
        check_dtype=False,
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {0: Double}


@pytest.mark.parametrize("null_type", ["pandas_na", "numpy_nan", "python_none"])
@pytest.mark.parametrize("test_case", ["boolean_with_null", "boolean_without_null"])
def test_imputer_bool_preserved(test_case, null_type):
    if test_case == "boolean_with_null":
        null_type = {"pandas_na": pd.NA, "numpy_nan": np.nan, "python_none": None}[
            null_type
        ]
        X = pd.DataFrame(pd.Series([True, False, True, null_type] * 4))
        X.ww.init(logical_types={0: BooleanNullable})
        expected = pd.DataFrame(
            pd.Series([True, False, True, True] * 4, dtype="bool"),
        )
    elif test_case == "boolean_without_null":
        X = pd.DataFrame(pd.Series([True, False, True, False] * 4))
        expected = pd.DataFrame(pd.Series([True, False, True, False] * 4))
    imputer = Imputer(categorical_impute_strategy="most_frequent")
    transformed = imputer.fit_transform(X)
    pd.testing.assert_frame_equal(
        transformed,
        expected,
    )
    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
        0: Boolean,
    }


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
    "data",
    [
        "int col",
        "float col",
        "categorical col",
        "bool col",
    ],
)
@pytest.mark.parametrize(
    "logical_type",
    ["Integer", "Double", "Categorical", "NaturalLanguage", "Boolean"],
)
@pytest.mark.parametrize("has_nan", ["has_nan", "no_nans"])
@pytest.mark.parametrize("numeric_impute_strategy", ["mean", "median", "most_frequent"])
def test_imputer_woodwork_custom_overrides_returned_by_components(
    data,
    logical_type,
    has_nan,
    numeric_impute_strategy,
    imputer_test_data,
):
    X_df = {
        "int col": imputer_test_data[["int col"]],
        "float col": imputer_test_data[["float col"]],
        "categorical col": imputer_test_data[["categorical col"]],
        "bool col": imputer_test_data[["bool col"]],
    }[data]
    logical_type = {
        "Integer": Double,
        "Double": Double,
        "Categorical": Categorical,
        "NaturalLanguage": NaturalLanguage,
        "Boolean": Boolean,
    }[logical_type]
    if has_nan == "has_nan" and logical_type == Boolean:
        logical_type = BooleanNullable
    y = pd.Series([1, 2, 1])
    try:
        X = X_df.copy()
        if has_nan == "has_nan" and logical_type == BooleanNullable:
            X.iloc[len(X_df) - 1, 0] = None
        elif has_nan == "has_nan":
            X.iloc[len(X_df) - 1, 0] = np.nan
        X.ww.init(logical_types={data: logical_type})
    except ww.exceptions.TypeConversionError:
        return

    imputer = Imputer(numeric_impute_strategy=numeric_impute_strategy)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert isinstance(transformed, pd.DataFrame)
    if logical_type == BooleanNullable:
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            data: Boolean,
        }
    elif numeric_impute_strategy == "most_frequent":
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            data: logical_type,
        }
    elif logical_type in [Categorical, NaturalLanguage] or has_nan == "no_nans":
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            data: logical_type,
        }
    else:
        assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
            data: Double,
        }
