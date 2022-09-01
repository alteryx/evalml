import numpy as np
import pandas as pd
import pytest
import woodwork as ww
from pandas.testing import assert_frame_equal
from sklearn.impute import KNNImputer as Sk_KNNImputer

from evalml.pipelines.components.transformers.imputers import KNNImputer
from evalml.utils.woodwork_utils import infer_feature_types


@pytest.mark.parametrize("n_neighbors", [1, 2, 5])
def test_knn_output(n_neighbors):
    X = pd.DataFrame(
        np.array(
            [
                [1, 1, 2],
                [1, 1, np.nan],
                [1, 1, 2],
                [1, 1, 2],
                [1, 2, 1],
                [1, 1, 2],
                [1, 1, 2],
            ],
        ),
    )
    sk_knn = Sk_KNNImputer(n_neighbors=n_neighbors)
    knn = KNNImputer(n_neighbors)
    assert_frame_equal(pd.DataFrame(sk_knn.fit_transform(X)), knn.fit_transform(X))


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

    X = infer_feature_types(X)
    assert_frame_equal(X.ww.types, X_t.ww.types)


@pytest.mark.parametrize("data_type", ["pd", "ww"])
def test_knn_imputer_boolean_dtype(data_type, make_data_type):
    X = pd.DataFrame(
        {
            "some_nan": pd.Series([True, np.nan, False, np.nan, True], dtype="boolean"),
        },
    )
    y = pd.Series([1, 0, 0, 1, 0])
    X_expected_arr = pd.DataFrame(
        {
            "some_nan": pd.Series([True, True, False, True, True]),
        },
    )
    X = make_data_type(data_type, X)
    imputer = KNNImputer(number_neighbors=1)
    X_t = imputer.fit_transform(X, y)

    assert type(X_t.ww.logical_types["some_nan"]) == ww.logical_types.Boolean
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
                dtype=bool,
            ),
            "bool no nan": pd.Series([False, False, False, False, True], dtype=bool),
        },
    )
    X_multi = make_data_type(data_type, X_multi)

    imputer = KNNImputer(number_neighbors=1)
    imputer.fit(X_multi, y)
    X_multi_t = imputer.transform(X_multi)
    assert_frame_equal(X_multi_expected_arr, X_multi_t)
    for col in X_multi_t:
        assert type(X_multi_t.ww.logical_types[col]) == ww.logical_types.Boolean


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
