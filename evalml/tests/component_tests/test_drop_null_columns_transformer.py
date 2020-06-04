import pandas as pd
import pytest

from evalml.pipelines.components import DropNullColumns


def test_drop_null_transformer_init():
    drop_null_transformer = DropNullColumns()
    assert drop_null_transformer.parameters["pct_null_threshold"] == 1.0
    assert drop_null_transformer.cols_to_drop is None

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.95)
    assert drop_null_transformer.parameters["pct_null_threshold"] == 0.95
    assert drop_null_transformer.cols_to_drop is None

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=-0.95)

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DropNullColumns(pct_null_threshold=1.01)


def test_drop_null_transformer_without_fit():
    drop_null_transformer = DropNullColumns()
    with pytest.raises(RuntimeError):
        drop_null_transformer.transform(pd.DataFrame())


def test_drop_null_transformer_transform():
    drop_null_transformer = DropNullColumns()
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'no_null': [1, 2, 3, 4, 5]})
    drop_null_transformer.fit(X)
    assert drop_null_transformer.transform(X).equals(X)

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

    drop_null_transformer = DropNullColumns(pct_null_threshold=0.0)
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                      'some_null': [None, 0, 3, 4, 5]})
    drop_null_transformer.fit(X)
    assert drop_null_transformer.transform(X).empty


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
