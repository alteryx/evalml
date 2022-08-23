import numpy as np
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


def test_knn_imputer_all_bool():
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
    imputer.fit(X, y)
    X_t = imputer.transform(X)
    X_expected = pd.DataFrame(
        {
            "Booleans": pd.Series(
                [True, True, True, False, False],
                dtype="bool",
            ),
        },
    )
    assert_frame_equal(X_expected, X_t)


def test_knn_imputer_revert_categorical_to_boolean():
    X = pd.DataFrame(
        {
            "Booleans": pd.Series(
                [True, True, True, False, False],
                dtype="boolean",
            ),
            "Numbers": pd.Series(
                [10, 11, 12, 13, 14],
                dtype="float",
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
            "Numbers": pd.Series(
                [10.0, 11.0, 12.0, 13.0, 14.0],
                dtype="float",
            ),
        },
    )
    assert_frame_equal(X_expected, X_t)


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
            ki = KNNImputer()
            ki.fit(X_df)
    else:
        ki = KNNImputer()
        ki.fit(X_df)
