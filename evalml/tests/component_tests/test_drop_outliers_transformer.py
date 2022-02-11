from unittest.mock import patch

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.data_checks import OutliersDataCheck
from evalml.pipelines.components.transformers.preprocessing import (
    DropOutliersTransformer,
)


def test_drop_outliers_transformer_init():
    drop_rows_transformer = DropOutliersTransformer()
    assert drop_rows_transformer.outlier_indices is None
    assert drop_rows_transformer.parameters == {}


@pytest.mark.parametrize("use_target", [True, False])
@patch("evalml.data_checks.OutliersDataCheck.get_outlier_rows")
def test_drop_outliers_transformer_fit_transform_with_target(
    mock_get_outlier_rows, use_target
):
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    X_expected = pd.DataFrame({"a column": [1], "another col": [4]})

    mock_get_outlier_rows.return_value = {"a column": [1, 2]}

    if use_target:
        y = pd.Series([1, 0, 1])
        y_expected = pd.Series([1])
    else:
        y = None
        y_expected = None

    drop_rows_transformer = DropOutliersTransformer()
    drop_rows_transformer.fit(X, y)
    transformed = drop_rows_transformer.transform(X, y)
    assert_frame_equal(X_expected, transformed[0])
    if use_target:
        assert_series_equal(y_expected, transformed[1])
    else:
        assert transformed[1] is None

    drop_rows_transformer = DropOutliersTransformer()
    fit_transformed = drop_rows_transformer.fit_transform(X, y)
    assert_frame_equal(fit_transformed[0], transformed[0])
    if use_target:
        assert_series_equal(y_expected, fit_transformed[1])
    else:
        assert fit_transformed[1] is None


@patch("evalml.data_checks.OutliersDataCheck.get_outlier_rows")
def test_drop_outliers_transformer_nonnumeric_index(mock_get_outlier_rows):
    X = pd.DataFrame({"numeric": [1, 2, 3], "cat": ["a", "b", "c"]})
    index = pd.Series(["i", "n", "d"])
    X = X.set_index(index)

    mock_get_outlier_rows.return_value = {"numeric": ["i"], "cat": ["n"]}

    X_expected = X.copy()
    X_expected.ww.init()
    X_expected = X_expected.drop(["i", "n"], axis=0)

    drop_rows_transformer = DropOutliersTransformer()
    drop_rows_transformer.fit(X)
    transformed = drop_rows_transformer.transform(X)
    assert_frame_equal(X_expected, transformed[0])
    assert transformed[1] is None

    y = pd.Series([1, 2, 3], index=index)
    y_expected = pd.Series([3], index=["d"])

    drop_rows_transformer = DropOutliersTransformer()
    drop_rows_transformer.fit(X, y)
    transformed = drop_rows_transformer.transform(X, y)
    assert_frame_equal(X_expected, transformed[0])
    assert_series_equal(y_expected, transformed[1])
