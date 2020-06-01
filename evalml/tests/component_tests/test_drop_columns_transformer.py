import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import DropColumnsTransformer


def test_drop_column_transformer_transform():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    drop_transformer = DropColumnsTransformer(columns=None)
    assert drop_transformer.transform(X).equals(X)

    drop_transformer = DropColumnsTransformer(columns=[])
    assert drop_transformer.transform(X).equals(X)

    drop_transformer = DropColumnsTransformer(columns=["one"])
    assert drop_transformer.transform(X).equals(X.loc[:, X.columns != "one"])

    drop_transformer = DropColumnsTransformer(columns=list(X.columns))
    assert drop_transformer.transform(X).empty


def test_drop_column_transformer_fit_transform():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    drop_transformer = DropColumnsTransformer(columns=[])
    assert drop_transformer.fit_transform(X).equals(X)
    assert drop_transformer.fit_transform(X).equals(drop_transformer.fit(X).transform(X))
    drop_transformer = DropColumnsTransformer(columns=["one"])
    assert drop_transformer.fit_transform(X).equals(X.loc[:, X.columns != "one"])

    drop_transformer = DropColumnsTransformer(columns=list(X.columns))
    assert drop_transformer.fit_transform(X).empty


def test_drop_column_transformer_input_invalid_col_name():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    drop_transformer = DropColumnsTransformer(columns=["not in data"])
    with pytest.raises(ValueError, match="Columns to drop do not exist in input data"):
        drop_transformer.transform(X)

    X = np.arange(12).reshape(3, 4)
    drop_transformer = DropColumnsTransformer(columns=[5])
    with pytest.raises(ValueError, match="Columns to drop do not exist in input data"):
        drop_transformer.transform(X)


def test_drop_column_transformer_indices():
    X = np.arange(12).reshape(3, 4)

    drop_transformer = DropColumnsTransformer(columns=[1])
    np.testing.assert_allclose(drop_transformer.transform(X).values, np.array([[0, 2, 3], [4, 6, 7], [8, 10, 11]]))

    drop_transformer = DropColumnsTransformer(columns=[0, 1, 2, 3])
    np.testing.assert_allclose(drop_transformer.transform(X).values, np.array([[], [], []]))
