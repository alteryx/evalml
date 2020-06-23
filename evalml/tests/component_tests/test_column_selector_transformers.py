import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import DropColumns, SelectColumns


def test_column_transformer_init():
    drop_transformer = DropColumns(columns=None)
    select_transformer = SelectColumns(columns=None)
    assert drop_transformer.parameters["columns"] is None
    assert select_transformer.parameters["columns"] is None

    drop_transformer = DropColumns(columns=[])
    select_transformer = SelectColumns(columns=[])
    assert drop_transformer.parameters["columns"] == []
    assert select_transformer.parameters["columns"] == []

    drop_transformer = DropColumns(columns=["a", "b"])
    select_transformer = SelectColumns(columns=["a", "b"])
    assert drop_transformer.parameters["columns"] == ["a", "b"]
    assert select_transformer.parameters["columns"] == ["a", "b"]

    with pytest.raises(ValueError, match="Parameter columns must be a list."):
        _ = DropColumns(columns="Column1")
    with pytest.raises(ValueError, match="Parameter columns must be a list."):
        _ = SelectColumns(columns="Column2")


def test_column_transformer_empty_X():
    X = pd.DataFrame()
    drop_transformer = DropColumns(columns=[])
    select_transformer = SelectColumns(columns=[])
    assert drop_transformer.transform(X).equals(X)
    assert select_transformer.transform(X).equals(X)

    drop_transformer = DropColumns(columns=[])
    select_transformer = SelectColumns(columns=[])
    assert drop_transformer.fit_transform(X).equals(X)
    assert select_transformer.fit_transform(X).equals(X)

    drop_transformer = DropColumns(columns=["not in data"])
    select_transformer = SelectColumns(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        drop_transformer.fit(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        select_transformer.fit(X)

    drop_transformer = DropColumns(columns=list(X.columns))
    select_transformer = SelectColumns(columns=list(X.columns))
    assert drop_transformer.transform(X).empty
    assert select_transformer.transform(X).empty


def test_column_transformer_transform():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    drop_transformer = DropColumns(columns=None)
    select_transformer = SelectColumns(columns=None)
    assert drop_transformer.transform(X).equals(X)
    assert select_transformer.transform(X).empty

    drop_transformer = DropColumns(columns=[])
    select_transformer = SelectColumns(columns=[])
    assert drop_transformer.transform(X).equals(X)
    assert select_transformer.transform(X).empty

    drop_transformer = DropColumns(columns=["one"])
    select_transformer = SelectColumns(columns=["one"])
    assert drop_transformer.transform(X).equals(X.drop(["one"], axis=1))
    assert select_transformer.transform(X).equals(X[["one"]])

    drop_transformer = DropColumns(columns=list(X.columns))
    select_transformer = SelectColumns(columns=list(X.columns))
    assert drop_transformer.transform(X).empty
    assert select_transformer.transform(X).equals(X)


def test_column_transformer_fit_transform():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    assert DropColumns(columns=[]).fit_transform(X).equals(X)
    assert SelectColumns(columns=[]).fit_transform(X).empty

    assert DropColumns(columns=["one"]).fit_transform(X).equals(X.drop(["one"], axis=1))
    assert DropColumns(columns=["one"]).fit_transform(X).equals(DropColumns(columns=["one"]).fit(X).transform(X))
    assert SelectColumns(columns=["one"]).fit_transform(X).equals(X[["one"]])

    assert DropColumns(columns=list(X.columns)).fit_transform(X).empty
    assert SelectColumns(columns=list(X.columns)).fit_transform(X).equals(X)


def test_drop_column_transformer_input_invalid_col_name():
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    drop_transformer = DropColumns(columns=["not in data"])
    select_transformer = SelectColumns(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        drop_transformer.fit(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        select_transformer.fit(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        drop_transformer.transform(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        select_transformer.transform(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        drop_transformer.fit_transform(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        select_transformer.fit_transform(X)

    X = np.arange(12).reshape(3, 4)
    drop_transformer = DropColumns(columns=[5])
    select_transformer = SelectColumns(columns=[5])
    with pytest.raises(ValueError, match="'5' not found in input data"):
        drop_transformer.fit(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        select_transformer.fit(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        drop_transformer.transform(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        select_transformer.transform(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        drop_transformer.fit_transform(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        select_transformer.fit_transform(X)


def test_column_transformer_numpy():
    X = np.arange(12).reshape(3, 4)

    drop_transformer = DropColumns(columns=[1])
    select_transformer = SelectColumns(columns=[1])
    np.testing.assert_allclose(drop_transformer.transform(X).values, np.array([[0, 2, 3], [4, 6, 7], [8, 10, 11]]))
    np.testing.assert_allclose(select_transformer.transform(X).values, np.array([[1], [5], [9]]))

    drop_transformer = DropColumns(columns=[0, 1, 2, 3])
    select_transformer = SelectColumns(columns=[0, 1, 2, 3])
    np.testing.assert_allclose(drop_transformer.transform(X).values, np.array([[], [], []]))
    np.testing.assert_allclose(select_transformer.transform(X).values, X)

    select_transformer = SelectColumns(columns=[])
    np.testing.assert_allclose(select_transformer.transform(X).values, np.array([[], [], []]))


def test_column_transformer_int_col_names():
    X = np.arange(12).reshape(3, 4)

    drop_transformer = DropColumns(columns=[1])
    select_transformer = SelectColumns(columns=[1])
    np.testing.assert_allclose(drop_transformer.transform(X).values,
                               pd.DataFrame(np.array([[0, 2, 3], [4, 6, 7], [8, 10, 11]])))
    np.testing.assert_allclose(select_transformer.transform(X).values, pd.DataFrame(np.array([[1], [5], [9]])))

    drop_transformer = DropColumns(columns=[0, 1, 2, 3])
    select_transformer = SelectColumns(columns=[0, 1, 2, 3])
    np.testing.assert_allclose(drop_transformer.transform(X).values, pd.DataFrame(np.array([[], [], []])))
    np.testing.assert_allclose(select_transformer.transform(X).values, X)
