import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from evalml.pipelines.components.transformers.preprocessing import DropRowsTransformer


def test_drop_rows_transformer_init():
    drop_rows_transformer = DropRowsTransformer()
    assert drop_rows_transformer.indices_to_drop is None

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[0, 1])
    assert drop_rows_transformer.indices_to_drop == [0, 1]
    assert drop_rows_transformer.parameters == {"indices_to_drop": [0, 1]}


def test_drop_rows_transformer_init_with_duplicate_indices():
    with pytest.raises(ValueError, match="All input indices must be unique."):
        DropRowsTransformer(indices_to_drop=[0, 0])


def test_drop_rows_transformer_fit_transform():
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    X_expected = X.copy()

    drop_rows_transformer_none = DropRowsTransformer()
    drop_rows_transformer_none.fit(X)
    transformed = drop_rows_transformer_none.transform(X)
    assert_frame_equal(X, transformed[0])
    assert transformed[1] is None

    indices_to_drop = [1, 2]
    X_expected = pd.DataFrame({"a column": [1], "another col": [4]})
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    drop_rows_transformer.fit(X)
    transformed = drop_rows_transformer.transform(X)
    assert_frame_equal(X_expected, transformed[0])
    assert transformed[1] is None

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    fit_transformed = drop_rows_transformer.fit_transform(X)
    assert_frame_equal(fit_transformed[0], transformed[0])
    assert fit_transformed[1] is None


def test_drop_rows_transformer_fit_transform_with_empty_indices_to_drop():
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    y = pd.Series([1, 0, 1])

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[])
    fit_transformed = drop_rows_transformer.fit_transform(X)
    assert_frame_equal(X, fit_transformed[0])
    assert fit_transformed[1] is None

    fit_transformed = drop_rows_transformer.fit_transform(X, y)
    assert_frame_equal(X, fit_transformed[0])
    assert_series_equal(y, fit_transformed[1])


def test_drop_rows_transformer_fit_transform_with_target():
    X = pd.DataFrame({"a column": [1, 2, 3], "another col": [4, 5, 6]})
    y = pd.Series([1, 0, 1])
    X_expected = pd.DataFrame({"a column": [1], "another col": [4]})
    y_expected = pd.Series([1])

    drop_rows_transformer_none = DropRowsTransformer()
    drop_rows_transformer_none.fit(X, y)
    transformed = drop_rows_transformer_none.transform(X, y)
    assert_frame_equal(X, transformed[0])
    assert_series_equal(y, transformed[1])

    indices_to_drop = [1, 2]
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    drop_rows_transformer.fit(X, y)
    transformed = drop_rows_transformer.transform(X, y)
    assert_frame_equal(X_expected, transformed[0])
    assert_series_equal(y_expected, transformed[1])

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    fit_transformed = drop_rows_transformer.fit_transform(X, y)
    assert_frame_equal(fit_transformed[0], transformed[0])
    assert_series_equal(y_expected, fit_transformed[1])


def test_drop_rows_transformer_index_not_in_input():
    X = pd.DataFrame({"numerical col": [1, 2]})
    y = pd.Series([0, 1], index=["a", "b"])
    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[100, 1])
    with pytest.raises(ValueError, match="do not exist in input features"):
        drop_rows_transformer.fit(X)

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=[0])
    with pytest.raises(ValueError, match="do not exist in input target"):
        drop_rows_transformer.fit(X, y)


def test_drop_rows_transformer_nonnumeric_index():
    X = pd.DataFrame({"numeric": [1, 2, 3], "cat": ["a", "b", "c"]})
    index = pd.Series(["i", "n", "d"])
    X = X.set_index(index)

    indices_to_drop = ["i", "n"]
    X_expected = X.copy()
    X_expected.ww.init()
    X_expected = X_expected.drop(indices_to_drop, axis=0)

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    drop_rows_transformer.fit(X)
    transformed = drop_rows_transformer.transform(X)
    assert_frame_equal(X_expected, transformed[0])
    assert transformed[1] is None

    y = pd.Series([1, 2, 3], index=index)
    y_expected = pd.Series([3], index=["d"])

    drop_rows_transformer = DropRowsTransformer(indices_to_drop=indices_to_drop)
    drop_rows_transformer.fit(X, y)
    transformed = drop_rows_transformer.transform(X, y)
    assert_frame_equal(X_expected, transformed[0])
    assert_series_equal(y_expected, transformed[1])
