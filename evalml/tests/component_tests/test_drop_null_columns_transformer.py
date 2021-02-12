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
    NaturalLanguage
)

from evalml.pipelines.components import DropNullColumns


def test_drop_null_transformer_init():
    drop_null_transformer = DropNullColumns(pct_null_threshold=0)
    assert drop_null_transformer.parameters == {"pct_null_threshold": 0.0}
    assert drop_null_transformer._cols_to_drop is None

    drop_null_transformer = DropNullColumns()
    assert drop_null_transformer.parameters == {"pct_null_threshold": 1.0}
    assert drop_null_transformer._cols_to_drop is None

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.95)
    assert drop_null_transformer.parameters == {"pct_null_threshold": 0.95}
    assert drop_null_transformer._cols_to_drop is None

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=-0.95)

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=1.01)


def test_drop_null_transformer_transform_default_pct_null_threshold():
    drop_null_transformer = DropNullColumns()
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'no_null': [1, 2, 3, 4, 5]})
    X_expected = X.astype({'lots_of_null': 'float64', 'no_null': 'Int64'})
    drop_null_transformer.fit(X)
    X_t = drop_null_transformer.transform(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())


def test_drop_null_transformer_transform_custom_pct_null_threshold():
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5]})

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    X_expected = X.drop(["lots_of_null", "all_null"], axis=1)
    X_expected = X_expected.astype({"no_null": "Int64"})
    drop_null_transformer.fit(X)
    X_t = drop_null_transformer.transform(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())
    # check that X is untouched
    assert X.equals(pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                                  'all_null': [None, None, None, None, None],
                                  'no_null': [1, 2, 3, 4, 5]}))


def test_drop_null_transformer_transform_boundary_pct_null_threshold():
    drop_null_transformer = DropNullColumns(pct_null_threshold=0.0)
    X = pd.DataFrame({'all_null': [None, None, None, None, None],
                      'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    drop_null_transformer.fit(X)
    X_t = drop_null_transformer.transform(X)
    assert X_t.to_dataframe().empty

    drop_null_transformer = DropNullColumns(pct_null_threshold=1.0)
    drop_null_transformer.fit(X)
    X_t = drop_null_transformer.transform(X)
    assert_frame_equal(X_t.to_dataframe(), X.drop(["all_null"], axis=1))
    # check that X is untouched
    assert X.equals(pd.DataFrame({'all_null': [None, None, None, None, None],
                                  'lots_of_null': [None, None, None, None, 5],
                                  'some_null': [None, 0, 3, 4, 5]}))


def test_drop_null_transformer_fit_transform():
    drop_null_transformer = DropNullColumns()
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'no_null': [1, 2, 3, 4, 5]})
    X_expected = X.astype({'lots_of_null': 'float64', 'no_null': 'Int64'})
    X_t = drop_null_transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())

    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5]})
    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    X_expected = X.drop(["lots_of_null", "all_null"], axis=1)
    X_expected = X_expected.astype({'no_null': 'Int64'})
    X_t = drop_null_transformer.fit_transform(X)
    assert_frame_equal(X_expected, X_t.to_dataframe())
    # check that X is untouched
    assert X.equals(pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                                  'all_null': [None, None, None, None, None],
                                  'no_null': [1, 2, 3, 4, 5]}))

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.0)
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    X_t = drop_null_transformer.fit_transform(X)
    assert X_t.to_dataframe().empty

    X = pd.DataFrame({'all_null': [None, None, None, None, None],
                      'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    drop_null_transformer = DropNullColumns(pct_null_threshold=1.0)
    X_t = drop_null_transformer.fit_transform(X)
    assert_frame_equal(X.drop(["all_null"], axis=1), X_t.to_dataframe())


def test_drop_null_transformer_np_array():
    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    X = np.array([[np.nan, 0, 2, 0],
                  [np.nan, 1, np.nan, 0],
                  [np.nan, 2, np.nan, 0],
                  [np.nan, 1, 1, 0]])
    X_t = drop_null_transformer.fit_transform(X)
    assert_frame_equal(X_t.to_dataframe(), pd.DataFrame(np.delete(X, [0, 2], axis=1), columns=[1, 3]))

    # check that X is untouched
    np.testing.assert_allclose(X, np.array([[np.nan, 0, 2, 0],
                                            [np.nan, 1, np.nan, 0],
                                            [np.nan, 2, np.nan, 0],
                                            [np.nan, 1, 1, 0]]))


@pytest.mark.parametrize("X_df", [pd.DataFrame(pd.Series([1, 2, 3], dtype="Int64")),
                                  pd.DataFrame(pd.Series([1., 2., 3.], dtype="float")),
                                  pd.DataFrame(pd.Series(['a', 'b', 'a'], dtype="category")),
                                  pd.DataFrame(pd.Series([True, False, True], dtype="boolean")),
                                  pd.DataFrame(pd.Series(['this will be a natural language column because length', 'yay', 'hay'], dtype="string"))])
@pytest.mark.parametrize("has_nan", [True, False])
def test_drop_null_transformer_woodwork_custom_overrides_returned_by_components(X_df, has_nan):
    y = pd.Series([1, 2, 1])
    if has_nan:
        X_df['all null'] = [np.nan, np.nan, np.nan]
    override_types = [Integer, Double, Categorical, NaturalLanguage, Boolean]
    for logical_type in override_types:
        try:
            X = ww.DataTable(X_df, logical_types={0: logical_type})
        except TypeError:
            continue

        drop_null_transformer = DropNullColumns()
        drop_null_transformer.fit(X)
        transformed = drop_null_transformer.transform(X, y)
        assert isinstance(transformed, ww.DataTable)
        assert transformed.logical_types == {0: logical_type}
