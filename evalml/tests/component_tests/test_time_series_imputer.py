import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from woodwork.logical_types import (
    AgeFractional,
    Boolean,
    BooleanNullable,
    Double,
    IntegerNullable,
)

from evalml.pipelines.components import TimeSeriesImputer


def test_invalid_strategy_parameters():
    with pytest.raises(ValueError, match="Valid numeric impute strategies are"):
        TimeSeriesImputer(numeric_impute_strategy="mean")
    with pytest.raises(ValueError, match="Valid categorical impute strategies are"):
        TimeSeriesImputer(categorical_impute_strategy="interpolate")
    with pytest.raises(ValueError, match="Valid target column impute strategies are"):
        TimeSeriesImputer(target_impute_strategy="not a strategy")


def test_imputer_default_parameters():
    imputer = TimeSeriesImputer()
    expected_parameters = {
        "categorical_impute_strategy": "forwards_fill",
        "numeric_impute_strategy": "interpolate",
        "target_impute_strategy": "forwards_fill",
    }
    assert imputer.parameters == expected_parameters


@pytest.mark.parametrize(
    "categorical_impute_strategy",
    ["forwards_fill", "backwards_fill"],
)
@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
@pytest.mark.parametrize(
    "target_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
def test_imputer_init(
    target_impute_strategy,
    numeric_impute_strategy,
    categorical_impute_strategy,
):
    imputer = TimeSeriesImputer(
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_impute_strategy=numeric_impute_strategy,
        target_impute_strategy=target_impute_strategy,
    )
    expected_parameters = {
        "categorical_impute_strategy": categorical_impute_strategy,
        "numeric_impute_strategy": numeric_impute_strategy,
        "target_impute_strategy": target_impute_strategy,
    }
    expected_hyperparameters = {
        "categorical_impute_strategy": ["backwards_fill", "forwards_fill"],
        "numeric_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
        "target_impute_strategy": ["backwards_fill", "forwards_fill", "interpolate"],
    }
    assert imputer.name == "Time Series Imputer"
    assert imputer.parameters == expected_parameters
    assert imputer.hyperparameter_ranges == expected_hyperparameters


def test_numeric_only_input(imputer_test_data):
    X = imputer_test_data[
        ["dates", "int col", "float col", "int with nan", "float with nan", "all nan"]
    ]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = TimeSeriesImputer(numeric_impute_strategy="backwards_fill")
    imputer.fit(X, y)
    transformed, _ = imputer.transform(X, y)
    expected = pd.DataFrame(
        {
            "dates": pd.date_range("01-01-2022", periods=20),
            "int col": [0, 1, 2, 0, 3] * 4,
            "float col": [0.1, 1.0, 0.0, -2.0, 5.0] * 4,
            "int with nan": [1, 1, 0, 0, 1] * 4,
            "float with nan": [0.3, 1.0, -1.0, -1.0, 0.0] * 4,
        },
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TimeSeriesImputer(numeric_impute_strategy="forwards_fill")
    imputer.fit(X, y)
    transformed, _ = imputer.transform(X, y)
    expected["float with nan"] = [0.3, 1.0, 1.0, -1.0, 0.0] * 4
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TimeSeriesImputer(numeric_impute_strategy="interpolate")
    imputer.fit(X, y)
    transformed, _ = imputer.fit_transform(X, y)
    expected["float with nan"] = [0.3, 1.0, 0.0, -1.0, 0.0] * 4
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_categorical_only_input(imputer_test_data):
    X = imputer_test_data[
        [
            "dates",
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
            "dates": pd.date_range("01-01-2022", periods=20),
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "bool col": [True, False, False, True, True] * 4,
            "categorical with nan": pd.Series(
                ["1", "1", "0", "0", "3"] + ["3", "1", "0", "0", "3"] * 3,
                dtype="category",
            ),
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "c"] * 4,
                dtype="category",
            ),
            "bool col with nan": pd.Series(
                [True, True, False, False, True] * 4,
                dtype="bool",
            ),
        },
    )
    imputer = TimeSeriesImputer()
    transformed, _ = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=True)
    assert "all nan cat" not in transformed.columns

    expected["categorical with nan"] = pd.Series(
        ["1", "1", "0", "0", "3"] * 4,
        dtype="category",
    )
    expected["object with nan"] = pd.Series(
        ["b", "b", "c", "c", "b"] * 3 + ["b", "b", "c", "c", "c"],
        dtype="category",
    )
    expected["bool col with nan"] = pd.Series(
        [True, False, False, True, True] * 4,
        dtype="bool",
    )

    imputer = TimeSeriesImputer(categorical_impute_strategy="backwards_fill")
    transformed, _ = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=True)


def test_categorical_and_numeric_input(imputer_test_data):
    X = imputer_test_data
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    # Copy X to avoid X taking on any mutations from the internal _handle_nullable_types call
    transformed, _ = imputer.transform(X.ww.copy(), y)
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
            "bool col 2": [True, False, False, True, True] * 4,
            "natural language col": pd.Series(
                ["cats are really great", "don't", "believe", "me?", "well..."] * 4,
                dtype="string",
            ),
            "categorical with nan": pd.Series(
                ["1", "1", "0", "0", "3"] + ["3", "1", "0", "0", "3"] * 3,
                dtype="category",
            ),
            "int with nan": [1, 1, 0, 0, 1] * 4,
            "float with nan": [0.3, 1.0, 0, -1.0, 0.0] * 4,
            "object with nan": pd.Series(
                ["b", "b", "b", "c", "c"] * 4,
                dtype="category",
            ),
            "bool col with nan": pd.Series(
                [True, True, False, False, True] * 4,
            ),
        },
    )
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TimeSeriesImputer(
        numeric_impute_strategy="forwards_fill",
        categorical_impute_strategy="forwards_fill",
    )
    # Copy X to avoid X taking on any mutations from the internal _handle_nullable_types call
    transformed, _ = imputer.fit_transform(X.ww.copy(), y)
    expected["float with nan"] = [0.3, 1.0, 1.0, -1.0, 0.0] * 4
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_impute_target():
    X = pd.DataFrame({"dates": pd.date_range("01-01-2022", periods=20)})
    y = pd.Series([0, 1, 0, 1, np.nan] * 4)
    y.ww.init(logical_type="double")

    imputer = TimeSeriesImputer(target_impute_strategy="forwards_fill")
    _, y_t = imputer.fit_transform(X, y)
    expected = pd.Series([0, 1, 0, 1, 1] * 4)
    assert_series_equal(y_t, expected, check_dtype=False)

    imputer = TimeSeriesImputer(target_impute_strategy="backwards_fill")
    _, y_t = imputer.fit_transform(X, y)
    expected = pd.Series([0, 1, 0, 1, 0] * 3 + [0, 1, 0, 1, 1])
    assert_series_equal(y_t, expected, check_dtype=False)

    imputer = TimeSeriesImputer(target_impute_strategy="interpolate")
    _, y_t = imputer.fit_transform(X, y)
    expected = pd.Series([0, 1, 0, 1, 0.5] * 3 + [0, 1, 0, 1, 1])
    assert_series_equal(y_t, expected, check_dtype=False)


def test_imputer_drops_columns_with_all_nan(imputer_test_data):
    X = imputer_test_data[["all nan cat", "all nan"]]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    X.ww.init()
    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    transformed, _ = imputer.transform(X, y)
    expected = X.drop(["all nan cat", "all nan"], axis=1)
    assert_frame_equal(transformed, expected, check_dtype=False)

    imputer = TimeSeriesImputer()
    transformed, _ = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, expected, check_dtype=False)


def test_typed_imputer_numpy_input():
    X = np.array([[1, 2, 2, 0], [np.nan, 0, 0, 0], [1, np.nan, np.nan, np.nan]])
    y = pd.Series([0, None, 1])
    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    X_t, y_t = imputer.transform(X, y)
    X_expected = pd.DataFrame(np.array([[1, 2, 2, 0], [1, 0, 0, 0], [1, 0, 0, 0]]))
    y_expected = pd.Series([0, 0, 1])
    assert_frame_equal(X_t, X_expected, check_dtype=False)
    assert_series_equal(y_t, y_expected, check_dtype=False)

    imputer = TimeSeriesImputer()
    X_t, y_t = imputer.fit_transform(X, y)
    assert_frame_equal(X_t, X_expected, check_dtype=False)
    assert_series_equal(y_t, y_expected, check_dtype=False)


@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_imputer_empty_data(data_type, make_data_type):
    X = pd.DataFrame()
    y = pd.Series()
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    X_expected = pd.DataFrame(index=pd.Index([]), columns=pd.Index([]))
    y_expected = pd.Series()

    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    X_t, y_t = imputer.transform(X, y)
    assert_frame_equal(X_t, X_expected, check_dtype=False)
    assert_series_equal(y_t, y_expected, check_dtype=False)

    imputer = TimeSeriesImputer()
    transformed, _ = imputer.fit_transform(X, y)
    assert_frame_equal(transformed, X_expected, check_dtype=False)


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
    X.ww.init(logical_types={"input_val": IntegerNullable, "input_cat": "categorical"})

    X.drop(0, inplace=True)
    y = X.ww.pop("target")

    expected = pd.DataFrame(
        {
            "input_val": [1.0, 2, 3, 4, 5, 6, 7, 8, 9],
            "input_cat": pd.Categorical(["a"] * 6 + ["b"] * 3),
        },
        index=list(range(1, 10)),
    )
    expected.ww.init(
        logical_types={"input_val": IntegerNullable, "input_cat": "categorical"},
    )

    imputer = TimeSeriesImputer()
    imputer.fit(X, y=y)
    transformed, _ = imputer.transform(X)
    expected = pd.DataFrame(
        {
            "input_val": [1, 2, 3, 4, 5, 6, 7, 8, 9],
            "input_cat": pd.Categorical(["a"] * 6 + ["b"] * 3),
        },
        index=list(range(1, 10)),
    ).astype({"input_val": float})

    pd.testing.assert_frame_equal(
        transformed,
        expected,
    )


def test_imputer_no_nans(imputer_test_data):
    X = imputer_test_data[
        ["categorical col", "int col", "object col", "float col", "bool col"]
    ]
    y = pd.Series([0, 0, 1, 0, 1] * 4)
    imputer = TimeSeriesImputer(
        categorical_impute_strategy="backwards_fill",
        numeric_impute_strategy="forwards_fill",
    )
    imputer.fit(X, y)
    X_t, y_t = imputer.transform(X, y)
    X_expected = pd.DataFrame(
        {
            "categorical col": pd.Series(
                ["zero", "one", "two", "zero", "two"] * 4,
                dtype="category",
            ),
            "int col": [0, 1, 2, 0, 3] * 4,
            "object col": pd.Series(["b", "b", "a", "c", "d"] * 4, dtype="category"),
            "float col": [0.1, 1.0, 0.0, -2.0, 5.0] * 4,
            "bool col": [True, False, False, True, True] * 4,
        },
    )
    assert_frame_equal(X_t, X_expected, check_dtype=False)
    assert_series_equal(y_t, y, check_dtype=False)


def test_imputer_with_none():
    X = pd.DataFrame(
        {
            "int with None": [1, 0, 5, None] * 4,
            "float with None": [0.1, 0.0, 0.5, None] * 4,
            "category with None": pd.Series(
                ["b", "a", "a", None] * 4,
                dtype="category",
            ),
            "boolean with None": pd.Series([True, None, False, True] * 4),
            "object with None": ["b", "a", "a", None] * 4,
            "all None": [None, None, None, None] * 4,
        },
    )
    X.ww.init(
        logical_types={
            "int with None": IntegerNullable,
            "float with None": "double",
            "category with None": "categorical",
            "boolean with None": BooleanNullable,
            "object with None": "categorical",
            "all None": "unknown",
        },
    )
    y = pd.Series([0, None, 1, 0, 1] * 4)
    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    X_t, y_t = imputer.transform(X, y)
    X_expected = pd.DataFrame(
        {
            "int with None": [1, 0, 5, 3] * 3 + [1, 0, 5, 5],
            "float with None": [0.1, 0.0, 0.5, 0.3] * 3 + [0.1, 0.0, 0.5, 0.5],
            "category with None": pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
            "boolean with None": pd.Series(
                [True, True, False, True] * 4,
                dtype="boolean",
            ),
            "object with None": pd.Series(["b", "a", "a", "a"] * 4, dtype="category"),
        },
    )
    y_expected = pd.Series([0, 0, 1, 0, 1] * 4)
    assert_frame_equal(X_expected, X_t, check_dtype=False)
    assert_series_equal(y_expected, y_t, check_dtype=False)

    imputer = TimeSeriesImputer()
    transformed, _ = imputer.fit_transform(X, y)
    assert_frame_equal(X_expected, transformed, check_dtype=False)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_all_bool_return_original(data_type, make_data_type):
    X = make_data_type(
        data_type,
        pd.DataFrame([True, True, False, True, True], dtype=bool),
    )
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype=bool)
    y = make_data_type(data_type, pd.Series([1, 0, 0, 1, 0]))

    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    X_t, _ = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_imputer_bool_dtype_object(data_type, make_data_type):
    X = pd.DataFrame([True, np.nan, False, np.nan, True] * 4)
    X.ww.init(logical_types={0: BooleanNullable})
    y = pd.Series([1, 0, 0, 1, 0] * 4)
    X_expected_arr = pd.DataFrame(
        [True, True, False, False, True] * 4,
        dtype="bool",
    )
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    imputer = TimeSeriesImputer()
    imputer.fit(X, y)
    X_t, _ = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_imputer_unusual_types_as_categorical():
    X = pd.DataFrame(
        {
            "urls": ["google.com", "github.com", None, "evalml.alteryx.com"] * 4,
            "emails": [None, "neil@gmail.com", "sally@aol.com", "yuri@university.edu"]
            * 4,
        },
    )
    X.ww.init(logical_types={"urls": "url"})
    y = pd.Series()

    X_expected = pd.DataFrame(
        {
            "urls": [
                "google.com",
                "github.com",
                "evalml.alteryx.com",
                "evalml.alteryx.com",
            ]
            * 4,
            "emails": [
                "neil@gmail.com",
                "neil@gmail.com",
                "sally@aol.com",
                "yuri@university.edu",
            ]
            * 4,
        },
    )
    imputer = TimeSeriesImputer(categorical_impute_strategy="backwards_fill")
    imputer.fit(X, y)
    X_t, _ = imputer.transform(X)
    assert_frame_equal(X_expected, X_t, check_dtype=False, check_categorical=False)

    X["categorical"] = pd.Series(["a", "b", "c", None] * 4, dtype="category")
    X["numeric"] = pd.Series([2, np.NaN, 4, 5] * 4)
    X.ww.init(logical_types={"urls": "url", "numeric": IntegerNullable})

    X_expected = pd.DataFrame(
        {
            "urls": ["google.com", "github.com", "github.com", "evalml.alteryx.com"]
            * 4,
            "emails": [
                "neil@gmail.com",
                "neil@gmail.com",
                "sally@aol.com",
                "yuri@university.edu",
            ]
            + [
                "yuri@university.edu",
                "neil@gmail.com",
                "sally@aol.com",
                "yuri@university.edu",
            ]
            * 3,
            "categorical": pd.Series(["a", "b", "c", "c"] * 4, dtype="category"),
            "numeric": pd.Series([2, 3, 4, 5] * 4),
        },
    )
    imputer = TimeSeriesImputer(categorical_impute_strategy="forwards_fill")
    imputer.fit(X, y)
    X_t, _ = imputer.transform(X)
    assert_frame_equal(X_expected, X_t, check_dtype=False, check_categorical=False)


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
                [True, True, False, False, False] * 4,
                dtype="bool",
            ),
            "bool no nan": pd.Series(
                [False, False, False, False, True] * 4,
                dtype=bool,
            ),
        },
    )

    X_multi = make_data_type(data_type, X_multi)
    y = make_data_type(data_type, y)

    imputer = TimeSeriesImputer()
    imputer.fit(X_multi, y)
    X_multi_t, _ = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t, check_dtype=False)


@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
def test_imputer_woodwork_custom_overrides_returned_by_components(
    numeric_impute_strategy,
    imputer_test_data,
):
    X = imputer_test_data
    y = pd.Series()

    X.ww.init(
        logical_types={
            "int with nan": "postal_code",
            "categorical with nan": "natural_language",
        },
    )

    imputer = TimeSeriesImputer(numeric_impute_strategy=numeric_impute_strategy)
    imputer.fit(X, y)
    transformed, _ = imputer.transform(X, y)
    assert isinstance(transformed, pd.DataFrame)
    assert (
        transformed.ww.logical_types["int with nan"]
        == X.ww.logical_types["int with nan"]
    )
    assert (
        transformed.ww.logical_types["categorical with nan"]
        == X.ww.logical_types["categorical with nan"]
    )


@pytest.mark.parametrize(
    "nullable_y_ltype, expected_imputed_y_ltype",
    [
        ("BooleanNullable", Double),
        ("IntegerNullable", Double),
        ("AgeNullable", AgeFractional),
    ],
)
@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
def test_imputer_can_take_in_nullable_types(
    nullable_type_test_data,
    nullable_type_target,
    numeric_impute_strategy,
    nullable_y_ltype,
    expected_imputed_y_ltype,
):
    y = nullable_type_target(ltype=nullable_y_ltype, has_nans=True)
    X = nullable_type_test_data(has_nans=True)
    # Drop the fully null columns since aren't relevant to the handle nullable types checks
    X = X.ww.drop(["all nan", "all nan cat"])

    cols_expected_to_change = X.ww.schema._filter_cols(
        include=["IntegerNullable", "AgeNullable", "BooleanNullable"],
    )
    cols_expected_to_stay_the_same = X.ww.schema._filter_cols(
        exclude=["IntegerNullable", "AgeNullable", "BooleanNullable"],
    )

    imputer = TimeSeriesImputer(
        numeric_impute_strategy=numeric_impute_strategy,
        target_impute_strategy="interpolate",
    )
    # Copy X to avoid X taking on any mutations from the internal _handle_nullable_types call
    imputer.fit(X.ww.copy(), y)
    X_imputed, y_imputed = imputer.transform(X.ww.copy(), y)

    assert not X_imputed.isnull().any().any()
    assert not y_imputed.isnull().any()

    # Check that the types are as expected - when interpolate is used, we need fractional numeric ltypes
    if numeric_impute_strategy == "interpolate":
        expected_X_ltypes = {"AgeFractional", "Double", "Boolean"}
    else:
        expected_X_ltypes = {"Age", "Integer", "Boolean"}

    assert X.ww.get_subset_schema(
        cols_expected_to_stay_the_same,
    ) == X_imputed.ww.get_subset_schema(cols_expected_to_stay_the_same)
    assert {
        str(ltype)
        for col, ltype in X_imputed.ww.logical_types.items()
        if col in cols_expected_to_change
    } == expected_X_ltypes

    assert isinstance(y_imputed.ww.logical_type, expected_imputed_y_ltype)


@pytest.mark.parametrize(
    "categorical_impute_strategy",
    ["forwards_fill", "backwards_fill"],
)
@pytest.mark.parametrize(
    "numeric_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
@pytest.mark.parametrize(
    "target_impute_strategy",
    ["forwards_fill", "backwards_fill", "interpolate"],
)
def test_imputer_nullable_handling_noop_for_non_impute_methods(
    nullable_type_test_data,
    nullable_type_target,
    target_impute_strategy,
    numeric_impute_strategy,
    categorical_impute_strategy,
):
    imputer = TimeSeriesImputer(
        categorical_impute_strategy=categorical_impute_strategy,
        numeric_impute_strategy=numeric_impute_strategy,
        target_impute_strategy=target_impute_strategy,
    )

    X = nullable_type_test_data(has_nans=True)
    y = nullable_type_target(ltype="IntegerNullable", has_nans=True)

    imputer.fit(X, y)
    original_X_schema = X.ww.schema
    original_y_schema = y.ww.schema
    X_d, y_d = imputer._handle_nullable_types(X, y)

    # Confirm that we only change inputs when interpolate is used
    if numeric_impute_strategy != "interpolate":
        assert X_d.ww.schema == original_X_schema
    else:
        assert X_d.ww.schema != original_X_schema

    if target_impute_strategy != "interpolate":
        assert y_d.ww.schema == original_y_schema
    else:
        assert y_d.ww.schema != original_y_schema


@pytest.mark.parametrize(
    "nullable_ltype",
    ["BooleanNullable", "IntegerNullable", "AgeNullable"],
)
@pytest.mark.parametrize(
    "handle_incompatibility",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(strict=True, raises=ValueError),
        ),
    ],
)
def test_time_series_imputer_nullable_type_incompatibility(
    nullable_type_target,
    handle_incompatibility,
    nullable_ltype,
):
    """Testing that the nullable type incompatibility that caused us to add handling for the time series imputer
    is still present in pandas' interpolate method. If this test is causing the test suite to fail
    because the code below no longer raises the expected ValueError, we should confirm that the nullable
    types now work for our use case and remove the nullable type handling logic from TimeSeriesImputer.
    """
    nullable_series = nullable_type_target(ltype=nullable_ltype, has_nans=True)
    if handle_incompatibility:
        imputer = TimeSeriesImputer(target_impute_strategy="interpolate")
        imputer.fit(pd.DataFrame(), nullable_series)
        _, nullable_series = imputer._handle_nullable_types(None, nullable_series)

    nullable_series.interpolate()
