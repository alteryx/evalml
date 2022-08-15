import numpy as np
import pandas
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from evalml.pipelines.components.transformers.imputers import KNNImputer


def test_knn_imputer_1_neighbor():
    X = pd.DataFrame(
        [
            [np.nan, 0, 1, np.nan],
            [1, 2, 3, 2],
            [10, 2, np.nan, 2],
            [10, 2, 5, np.nan],
            [6, 2, 7, 0],
            [1, 2, 3, 4],
            [1, np.nan, 3, 2],
        ],
    )
    transformer = KNNImputer(number_neighbors=1)
    X_expected_arr = pd.DataFrame(
        [
            [1, 0, 1, 2],
            [1, 2, 3, 2],
            [10, 2, 5, 2],
            [10, 2, 5, 2],
            [6, 2, 7, 0],
            [1, 2, 3, 4],
            [1, 2, 3, 2],
        ],
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


def test_knn_imputer_2_neighbors():
    X = pd.DataFrame([[np.nan, 0, 3, np.nan], [1, 2, 3, 2], [1, 2, 3, 0]])
    # test impute_strategy
    transformer = KNNImputer(number_neighbors=2)
    X_expected_arr = pd.DataFrame(
        [[1, 0, 3, 1], [1, 2, 3, 2], [1, 2, 3, 0]],
    )
    X_t = transformer.fit_transform(X)
    assert_frame_equal(X_expected_arr, X_t, check_dtype=False)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_knn_imputer_all_bool_return_original(data_type, make_data_type):
    X = pd.DataFrame([True, True, False, True, True], dtype=bool)
    y = pd.Series([1, 0, 0, 1, 0])
    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype=bool)
    imputer = KNNImputer()
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_knn_imputer_boolean_dtype(data_type, make_data_type):
    X = pd.DataFrame([True, np.nan, False, np.nan, True])
    X.ww.init(logical_types={0: "BooleanNullable"})
    y = pd.Series([1, 0, 0, 1, 0])
    X_expected_arr = pd.DataFrame([True, True, False, True, True], dtype="boolean")
    X = make_data_type(data_type, X)
    print(X)
    imputer = KNNImputer(number_neighbors=1)
    X_t = imputer.fit_transform(X, y)
    assert_frame_equal(X_expected_arr, X_t)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_knn_imputer_multitype_with_one_bool(data_type, make_data_type):
    X_multi = pd.DataFrame(
        {
            "bool with nan": pd.Series([True, np.nan, False, np.nan, False]),
            "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
        },
    )
    X_multi.ww.init(logical_types={"bool with nan": "BooleanNullable"})
    y = pd.Series([1, 0, 0, 1, 0])
    X_multi_expected_arr = pd.DataFrame(
        {
            "bool with nan": pd.Series(
                [True, True, False, True, False],
                dtype="boolean",
            ),
            "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
        },
    )
    X_multi = make_data_type(data_type, X_multi)

    imputer = KNNImputer(number_neighbors=1)
    imputer.fit(X_multi, y)
    X_multi_t = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t)


def test_knn_imputer_revert_categorical_to_boolean():
    X = pd.DataFrame(
        {
            "Booleans": pd.Series(
                [True, True, True, False, False],
                dtype="boolean",
            ),
        },
    )
    y = pd.Series([1, 1, 1, 0, 0])
    imputer = KNNImputer(number_neighbors=1)
    X_t = imputer.fit_transform(X, y)
    X_expected = pd.DataFrame(
        {
            "Booleans": pd.Series(
                [True, True, True, False, False],
                dtype="bool",
            ),
        },
    )
    assert_frame_equal(X_expected, X_t)


'''


def test_simple_imputer_transform_drop_all_nan_columns_empty():
    X = pd.DataFrame([[np.nan, np.nan, np.nan]])
    X.ww.init(logical_types={0: "Double", 1: "Double", 2: "Double"})
    transformer = SimpleImputer(impute_strategy="most_frequent")
    assert transformer.fit_transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))

    transformer = SimpleImputer(impute_strategy="most_frequent")
    transformer.fit(X)
    assert transformer.transform(X).empty
    assert_frame_equal(X, pd.DataFrame([[np.nan, np.nan, np.nan]]))


def test_simple_imputer_numpy_input():
    X = np.array([[1, 0, 1, np.nan], [np.nan, 2, 3, 2], [np.nan, 2, 3, 0]])
    transformer = SimpleImputer(impute_strategy="mean")
    X_expected_arr = np.array([[1, 0, 1, 1], [1, 2, 3, 2], [1, 2, 3, 0]])
    assert np.allclose(X_expected_arr, transformer.fit_transform(X))
    np.testing.assert_almost_equal(
        X,
        np.array([[1, 0, 1, np.nan], [np.nan, 2, 3, 2], [np.nan, 2, 3, 0]]),
    )


@pytest.mark.parametrize("data_type", ["numeric", "categorical"])
def test_simple_imputer_fill_value(data_type):
    if data_type == "numeric":
        X = pd.DataFrame(
            {"some numeric": [np.nan, 1, 0], "another numeric": [0, np.nan, 2]},
        )
        fill_value = -1
        expected = pd.DataFrame(
            {"some numeric": [fill_value, 1, 0], "another numeric": [0, fill_value, 2]},
        )
    else:
        X = pd.DataFrame(
            {
                "categorical with nan": pd.Series(
                    [np.nan, "1", np.nan, "0", "3"],
                    dtype="category",
                ),
                "object with nan": ["b", "b", np.nan, "c", np.nan],
            },
        )
        fill_value = "fill"
        expected = pd.DataFrame(
            {
                "categorical with nan": pd.Series(
                    ["fill", "1", "fill", "0", "3"],
                    dtype="category",
                ),
                "object with nan": pd.Series(
                    ["b", "b", "fill", "c", "fill"],
                    dtype="category",
                ),
            },
        )
        X.ww.init(
            logical_types={
                "categorical with nan": "categorical",
                "object with nan": "categorical",
            },
        )
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = SimpleImputer(impute_strategy="constant", fill_value=fill_value)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert_frame_equal(expected, transformed, check_dtype=False)

    imputer = SimpleImputer(impute_strategy="constant", fill_value=fill_value)
    transformed = imputer.fit_transform(X, y)
    assert_frame_equal(expected, transformed, check_dtype=False)


def test_simple_imputer_does_not_reset_index():
    X = pd.DataFrame({"input_val": np.arange(10), "target": np.arange(10)})
    X.loc[5, "input_val"] = np.nan
    assert X.index.tolist() == list(range(10))

    X.drop(0, inplace=True)
    y = X.pop("target")
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            {"input_val": [1.0, 2, 3, 4, np.nan, 6, 7, 8, 9]},
            dtype=float,
            index=list(range(1, 10)),
        ),
        X,
    )

    imputer = SimpleImputer(impute_strategy="mean")
    imputer.fit(X, y=y)
    transformed = imputer.transform(X)
    pd.testing.assert_frame_equal(
        pd.DataFrame(
            {"input_val": [1, 2, 3, 4, 5, 6, 7, 8, 9]},
            dtype=float,
            index=list(range(1, 10)),
        ),
        transformed,
    )


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
        "numerics_only",
        "booleans_only",
        "categoricals_only",
        pytest.param(
            "categorical_and_booleans",
            marks=pytest.mark.xfail(
                reason="Since the scikit-learn 1.1 upgrade, SimpleImputer can't deal with categoricals and booleans in same array",
            ),
        ),
        pytest.param(
            "all",
            marks=pytest.mark.xfail(
                reason="Since the scikit-learn 1.1 upgrade, SimpleImputer can't deal with categoricals and booleans in same array",
            ),
        ),
    ],
)
def test_simple_imputer_with_none_separated(dtypes):
    test_ltypes = dict((k, ltypes[k]) for k in columns_dict[dtypes])
    X_test = X[columns_dict[dtypes]]
    X_test.ww.init(logical_types=test_ltypes)
    y = pd.Series([0, 0, 1, 0, 1])
    imputer = SimpleImputer()
    imputer.fit(X_test, y)
    transformed = imputer.transform(X_test, y)
    assert_frame_equal(expected[columns_dict[dtypes]], transformed, check_dtype=False)


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
@pytest.mark.parametrize("impute_strategy", ["mean", "median"])
def test_simple_imputer_woodwork_custom_overrides_returned_by_components(
    data,
    logical_type,
    has_nan,
    impute_strategy,
    imputer_test_data,
):
    X_df = {
        "int col": imputer_test_data[["int col"]],
        "float col": imputer_test_data[["float col"]],
        "categorical col": imputer_test_data[["categorical col"]],
        "bool col": imputer_test_data[["bool col"]],
    }[data]
    logical_type = {
        "Integer": Integer,
        "Double": Double,
        "Categorical": Categorical,
        "NaturalLanguage": NaturalLanguage,
        "Boolean": Boolean,
    }[logical_type]
    y = pd.Series([1, 2, 1])

    # Categorical -> Boolean fails in infer_feature_types
    if data == "categorical col" and logical_type == Boolean:
        return
    try:
        X = X_df.copy()
        if has_nan == "has_nan":
            X.iloc[len(X_df) - 1, 0] = np.nan
        X.ww.init(logical_types={data: logical_type})
    except ww.exceptions.TypeConversionError:
        return

    impute_strategy_to_use = impute_strategy
    if logical_type in [NaturalLanguage, Categorical]:
        impute_strategy_to_use = "most_frequent"

    imputer = SimpleImputer(impute_strategy=impute_strategy_to_use)
    imputer.fit(X, y)
    transformed = imputer.transform(X, y)
    assert isinstance(transformed, pd.DataFrame)

    assert {k: type(v) for k, v in transformed.ww.logical_types.items()} == {
        data: logical_type,
    }


def test_component_handles_pre_init_ww():
    """Test to determine whether SimpleImputer can handle
    a Woodwork-inited DataFrame with partially null and fully
    null columns (post Woodwork 0.5.1) and still perform the
    expected behavior."""
    df = pd.DataFrame(
        {"part_null": [0, 1, 2, None], "all_null": [None, None, None, None]},
    )
    df.ww.init(logical_types={"all_null": "Double"})
    imputed = SimpleImputer().fit_transform(df)

    assert "all_null" not in imputed.columns
    assert [x for x in imputed["part_null"]] == [0, 1, 2, 0]
'''


@pytest.mark.parametrize("df_composition", ["full_df", "single_column"])
@pytest.mark.parametrize("has_nan", ["has_nan", "no_nans"])
def test_knn_imputer_ignores_natural_language(
    has_nan,
    imputer_test_data,
    df_composition,
):
    """Test to ensure that the simple imputer just passes through
    natural language columns, unchanged."""
    if df_composition == "single_column":
        X_df = imputer_test_data[["natural language col"]]
        X_df.ww.init()
    elif df_composition == "full_df":
        X_df = imputer_test_data[["int col", "float col", "natural language col"]]
        X_df.ww.init()

    if has_nan == "has_nan":
        X_df.iloc[-1, :] = None
        X_df.ww.init()
    y = pd.Series([x for x in range(X_df.shape[1])])

    imputer = KNNImputer(number_neighbors=3)

    imputer.fit(X_df, y)

    result = imputer.transform(X_df, y)

    # raise Exception
    if df_composition == "full_df":
        X_df = X_df.astype(
            {"int col": float},
        )  # Convert to float as the imputer will do this as we're requesting KNN
        result = result.astype(
            {"int col": float},
        )
        X_df["float col"] = result["float col"]
        X_df["int col"] = result["int col"]
        assert_frame_equal(result, X_df)
    elif df_composition == "single_column":
        assert_frame_equal(result, X_df)


@pytest.mark.parametrize(
    "data",
    [
        ["int col"],
        ["float col"],
        ["categorical col", "bool col"],
        ["bool col", "float col"],
    ],
)
def test_knn_imputer_errors_with_bool_and_categorical_columns(
    data,
    imputer_test_data,
):
    X_df = imputer_test_data[data]
    if "categorical col" in data and "bool col" in data:
        with pytest.raises(
            ValueError,
            match="KNNImputer cannot handle dataframes with both boolean and categorical features.",
        ):
            si = KNNImputer()
            si.fit(X_df)
    else:
        si = KNNImputer()
        si.fit(X_df)
