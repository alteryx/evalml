import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from woodwork import init_series

from evalml.pipelines.components.transformers.preprocessing import (
    DropNaNRowsTransformer,
)
from evalml.utils.woodwork_utils import _schema_is_equal


def test_drop_rows_transformer():
    # Expecting float because of np.NaN values
    X = pd.DataFrame({"a column": [np.NaN, 2, 3], "another col": [4, np.NaN, 6]})
    X_expected = pd.DataFrame(
        {"a column": [3], "another col": [6]},
        index=[2],
    ).astype("Int64")
    drop_rows_transformer = DropNaNRowsTransformer()
    drop_rows_transformer.fit(X)
    transformed_X, _ = drop_rows_transformer.transform(X)
    assert_frame_equal(transformed_X, X_expected)

    drop_rows_transformer = DropNaNRowsTransformer()
    fit_transformed_X, _ = drop_rows_transformer.fit_transform(X)
    assert_frame_equal(fit_transformed_X, X_expected)


@pytest.mark.parametrize("null_value", [pd.NA, np.NaN])
def test_drop_rows_transformer_retain_ww_schema(null_value):
    # Expecting float because of np.NaN values
    X = pd.DataFrame(
        {"a column": [null_value, 2, 3, 4], "another col": ["a", null_value, "c", "d"]},
    )
    X.ww.init()
    X.ww.set_types(
        logical_types={"a column": "IntegerNullable", "another col": "PersonFullName"},
        semantic_tags={"a column": "custom_tag"},
    )

    X_expected = pd.DataFrame({"a column": [3], "another col": ["c"]}, index=[2])
    X_expected.ww.init_with_partial_schema(
        logical_types={"a column": "IntegerNullable", "another col": "PersonFullName"},
        semantic_tags={"a column": "custom_tag"},
    )
    X_expected_schema = X.ww.schema

    y = pd.Series([3, 2, 1, null_value])
    y = init_series(y, logical_type="IntegerNullable", semantic_tags="y_custom_tag")

    y_expected = pd.Series([1], index=[2])
    y_expected = init_series(
        y_expected,
        logical_type="IntegerNullable",
        semantic_tags="y_custom_tag",
    )
    y_expected_schema = y.ww.schema

    drop_rows_transformer = DropNaNRowsTransformer()
    transformed_X, transformed_y = drop_rows_transformer.fit_transform(X, y)
    assert_frame_equal(transformed_X, X_expected)
    assert_series_equal(transformed_y, y_expected)
    assert _schema_is_equal(transformed_X.ww.schema, X_expected_schema)
    assert transformed_y.ww.schema == y_expected_schema
