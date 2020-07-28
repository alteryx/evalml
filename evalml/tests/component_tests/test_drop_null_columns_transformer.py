import numpy as np
import pandas as pd
import pytest

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
    drop_null_transformer.fit(X)
    assert drop_null_transformer.transform(X).equals(X)


def test_drop_null_transformer_transform_custom_pct_null_threshold():
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5]})

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    drop_null_transformer.fit(X)
    assert drop_null_transformer.transform(X).equals(X.drop(["lots_of_null", "all_null"], axis=1))
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
    assert drop_null_transformer.transform(X).empty

    drop_null_transformer = DropNullColumns(pct_null_threshold=1.0)
    drop_null_transformer.fit(X)
    assert drop_null_transformer.transform(X).equals(X.drop(["all_null"], axis=1))
    # check that X is untouched
    assert X.equals(pd.DataFrame({'all_null': [None, None, None, None, None],
                                  'lots_of_null': [None, None, None, None, 5],
                                  'some_null': [None, 0, 3, 4, 5]}))


def test_drop_null_transformer_fit_transform():
    drop_null_transformer = DropNullColumns()
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'no_null': [1, 2, 3, 4, 5]})
    assert drop_null_transformer.fit_transform(X).equals(X)

    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5]})

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    assert drop_null_transformer.fit_transform(X).equals(X.drop(["lots_of_null", "all_null"], axis=1))
    # check that X is untouched
    assert X.equals(pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                                  'all_null': [None, None, None, None, None],
                                  'no_null': [1, 2, 3, 4, 5]}))

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.0)
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    assert drop_null_transformer.fit_transform(X).empty

    X = pd.DataFrame({'all_null': [None, None, None, None, None],
                      'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    drop_null_transformer = DropNullColumns(pct_null_threshold=1.0)
    assert drop_null_transformer.fit_transform(X).equals(X.drop(["all_null"], axis=1))


def test_drop_null_transformer_np_array():
    drop_null_transformer = DropNullColumns(pct_null_threshold=0.5)
    X = np.array([[np.nan, 0, 2, 0],
                  [np.nan, 1, np.nan, 0],
                  [np.nan, 2, np.nan, 0],
                  [np.nan, 1, 1, 0]])
    assert drop_null_transformer.fit_transform(X).equals(pd.DataFrame(np.delete(X, [0, 2], axis=1), columns=[1, 3]))

    # check that X is untouched
    np.testing.assert_allclose(X, np.array([[np.nan, 0, 2, 0],
                                            [np.nan, 1, np.nan, 0],
                                            [np.nan, 2, np.nan, 0],
                                            [np.nan, 1, 1, 0]]))
