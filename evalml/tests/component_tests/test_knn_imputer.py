import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from sklearn.impute import KNNImputer as Sk_KNNImputer

from evalml.pipelines.components.transformers.imputers import KNNImputer


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


@pytest.mark.parametrize("df_composition", ["full_df", "single_column"])
@pytest.mark.parametrize("has_nan", ["has_nan", "no_nans"])
def test_knn_imputer_ignores_natural_language(
    has_nan,
    imputer_test_data,
    df_composition,
):
    """Test to ensure that the knn imputer just passes through
    natural language columns, unchanged.
    """
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
        # Update the other columns in result so that we can confirm the natural language column
        # is unchanged
        X_df["float col"] = result["float col"]
        X_df["int col"] = result["int col"]
        assert_frame_equal(result, X_df)
    elif df_composition == "single_column":
        assert_frame_equal(result, X_df)


def test_knn_imputer_maintains_woodwork_types(imputer_test_data):
    X = imputer_test_data.ww.select("numeric")
    int_nullable_cols = X.ww.select("IntegerNullable").columns.to_list()
    unchanged_schema = X.ww.drop(int_nullable_cols).ww.schema
    y = pd.Series([x for x in range(X.shape[1])])

    imputer = KNNImputer(number_neighbors=3)

    imputer.fit(X, y)
    result = imputer.transform(X, y)
    assert unchanged_schema == result.ww.drop(int_nullable_cols).ww.schema
    assert {
        str(ltype)
        for col, ltype in result.ww.logical_types.items()
        if col in int_nullable_cols
    } == {"Double"}


def test_knn_imputer_with_all_null_and_nl_cols(
    imputer_test_data,
):
    X = imputer_test_data.ww[["all nan", "natural language col", "int col"]]
    X_copy = X.ww.copy()

    imp = KNNImputer(number_neighbors=3)
    imp.fit(X)

    X_imputed = imp.transform(X)
    pd.testing.assert_frame_equal(X_copy.ww.drop("all nan"), X_imputed)
