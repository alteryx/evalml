import numpy as np
import pandas as pd
import pytest

from evalml.pipelines.components import DropColumns, SelectColumns


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_column_transformer_init(class_to_test):
    transformer = class_to_test(columns=None)
    assert transformer.parameters["columns"] is None

    transformer = class_to_test(columns=[])
    assert transformer.parameters["columns"] == []

    transformer = class_to_test(columns=["a", "b"])
    assert transformer.parameters["columns"] == ["a", "b"]

    with pytest.raises(ValueError, match="Parameter columns must be a list."):
        _ = class_to_test(columns="Column1")


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_column_transformer_empty_X(class_to_test):
    X = pd.DataFrame()
    transformer = class_to_test(columns=[])
    assert transformer.transform(X).equals(X)

    transformer = class_to_test(columns=[])
    assert transformer.fit_transform(X).equals(X)

    transformer = class_to_test(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.fit(X)

    transformer = class_to_test(columns=list(X.columns))
    assert transformer.transform(X).empty


@pytest.mark.parametrize("class_to_test,checking_functions",
                         [(DropColumns, [lambda df, X: df.equals(X),
                                         lambda df, X: df.equals(X),
                                         lambda df, X: df.equals(X.drop(columns=["one"])),
                                         lambda df, X: df.empty]),
                          (SelectColumns, [lambda df, X: df.empty,
                                           lambda df, X: df.empty,
                                           lambda df, X: df.equals(X[["one"]]),
                                           lambda df, X: df.equals(X)])
                          ])
def test_column_transformer_transform(class_to_test, checking_functions):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    check1, check2, check3, check4 = checking_functions

    transformer = class_to_test(columns=None)
    assert check1(transformer.transform(X), X)

    transformer = class_to_test(columns=[])
    assert check2(transformer.transform(X), X)

    transformer = class_to_test(columns=["one"])
    assert check3(transformer.transform(X), X)

    transformer = class_to_test(columns=list(X.columns))
    assert check4(transformer.transform(X), X)


@pytest.mark.parametrize("class_to_test,checking_functions",
                         [(DropColumns, [lambda df, X: df.equals(X),
                                         lambda df, X: df.equals(X.drop(columns=["one"])),
                                         lambda df, X: df.empty]),
                          (SelectColumns, [lambda df, X: df.empty,
                                           lambda df, X: df.equals(X[["one"]]),
                                           lambda df, X: df.equals(X)])
                          ])
def test_column_transformer_fit_transform(class_to_test, checking_functions):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    check1, check2, check3 = checking_functions

    assert check1(class_to_test(columns=[]).fit_transform(X), X)

    assert check2(class_to_test(columns=["one"]).fit_transform(X), X)

    assert check3(class_to_test(columns=list(X.columns)).fit_transform(X), X)


@pytest.mark.parametrize("class_to_test", [DropColumns, SelectColumns])
def test_drop_column_transformer_input_invalid_col_name(class_to_test):
    X = pd.DataFrame({'one': [1, 2, 3, 4], 'two': [2, 3, 4, 5], 'three': [1, 2, 3, 4]})
    transformer = class_to_test(columns=["not in data"])
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.fit(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.transform(X)
    with pytest.raises(ValueError, match="'not in data' not found in input data"):
        transformer.transform(X)

    X = np.arange(12).reshape(3, 4)
    transformer = class_to_test(columns=[5])
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.fit(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.transform(X)
    with pytest.raises(ValueError, match="'5' not found in input data"):
        transformer.fit_transform(X)


@pytest.mark.parametrize("class_to_test,answers",
                         [(DropColumns, [np.array([[0, 2, 3], [4, 6, 7], [8, 10, 11]]),
                                         np.array([[], [], []]),
                                         np.arange(12).reshape(3, 4)]),
                          (SelectColumns, [np.array([[1], [5], [9]]),
                                           np.arange(12).reshape(3, 4),
                                           np.array([[], [], []])])
                          ])
def test_column_transformer_numpy(class_to_test, answers):
    X = np.arange(12).reshape(3, 4)
    answer1, answer2, answer3 = answers

    transformer = class_to_test(columns=[1])
    np.testing.assert_allclose(transformer.transform(X).values, answer1)

    transformer = class_to_test(columns=[0, 1, 2, 3])
    np.testing.assert_allclose(transformer.transform(X).values, answer2)

    transformer = class_to_test(columns=[])
    np.testing.assert_allclose(transformer.transform(X).values, answer3)


@pytest.mark.parametrize("class_to_test,answers",
                         [(DropColumns, [np.array([[0, 2, 3], [4, 6, 7], [8, 10, 11]]),
                                         np.array([[], [], []])]),
                          (SelectColumns, [np.array([[1], [5], [9]]),
                                           np.arange(12).reshape(3, 4)])
                          ])
def test_column_transformer_int_col_names(class_to_test, answers):
    X = np.arange(12).reshape(3, 4)
    answer1, answer2 = answers

    transformer = class_to_test(columns=[1])
    np.testing.assert_allclose(transformer.transform(X).values, answer1)

    transformer = class_to_test(columns=[0, 1, 2, 3])
    np.testing.assert_allclose(transformer.transform(X).values, answer2)
