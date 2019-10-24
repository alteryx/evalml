import numpy as np
import pandas as pd

from evalml.guardrails import (
    detect_categorical_correlation,
    detect_collinearity,
    detect_correlation,
    detect_multicollinearity
)


def test_detect_collinearity():
    col = pd.Series([1, 0, 2, 3, 4])
    X = pd.DataFrame()
    X["a"] = col * 3
    X["b"] = ~col
    X["c"] = col / 2
    X["d"] = col + 1
    X["e"] = [0, 1, 0, 0, 0]
    result = detect_collinearity(X)
    expected = {('a', 'b'), ('a', 'c'), ('a', 'd'), ('b', 'c'), ('b', 'd'), ('c', 'd')}
    assert expected == set(result.keys())


def test_detect_multicollinearity():
    col = pd.Series([1, 0, 2, 3, 4])
    X = pd.DataFrame()
    X["a"] = col * 3
    X["b"] = ~col
    X["c"] = col / 2
    X["d"] = col + 1
    X["e"] = [0, 1, 0, 0, 0]
    expected = set(["a", "b", "c", "d"])
    result = detect_multicollinearity(X)
    assert expected == set(result.keys())


def test_detect_multicollinearity_only():
    # test detect_multicollinearity on data that is only multicollinear (not collinear)
    col = np.random.randn(100, 10)
    noise = np.random.randn(100)
    col[:, 9] = 3 * col[:, 2] + 1.5 * col[:, 3] + 5 * col[:, 6] + .5 * noise
    X = pd.DataFrame(data=col)
    expected = set([2, 3, 6, 9])
    result = detect_multicollinearity(X)
    assert expected == set(result.keys())


def test_detect_categorical_correlation():
    X = {'col_1': [1, 1, 2, 3, 1, 2, 3, 4],
         'col_2': ['a', 'a', 'b', 'c', 'a', 'b', 'c', 'd'],
         'col_3': ['w', 'w', 'x', 'y', 'w', 'x', 'y', 'z'],
         'col_4': [1, 1, 4, 3, 1, 3, 3, 1]}
    X = pd.DataFrame(data=X)
    for col in X:
        X[col] = X[col].astype('category')

    result = detect_categorical_correlation(X, 0.9)
    expected = {('col_1', 'col_3'), ('col_1', 'col_2'), ('col_2', 'col_3')}
    assert expected == set(result.keys())


def test_detect_correlation():
    col = pd.Series([1, 0, 2, 3, 4, 0, 1, 5, 2, 1])
    X_num = pd.DataFrame()
    X_num["a"] = col * 3
    X_num["b"] = col / 2
    X_num["c"] = col + 1
    X_num["d"] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

    X_cat = {'col_1': [1, 1, 2, 3, 1, 2, 3, 4, 1, 3],
             'col_2': ['a', 'a', 'b', 'c', 'a', 'b', 'c', 'd', 'a', 'c'],
             'col_3': ['w', 'w', 'x', 'y', 'w', 'x', 'y', 'z', 'w', 'y'],
             'col_4': [1, 1, 4, 3, 1, 3, 3, 1, 2, 3]}
    X_cat = pd.DataFrame(data=X_cat)
    for col in X_cat:
        X_cat[col] = X_cat[col].astype('category')
    X = pd.concat([X_num, X_cat], axis=1)
    expected = set([('a', 'b'), ('a', 'c'), ('b', 'c'), ('col_1', 'col_2'), ('col_1', 'col_3'), ('col_2', 'col_3')])
    result = detect_correlation(X)
    assert expected == set(result.keys())
