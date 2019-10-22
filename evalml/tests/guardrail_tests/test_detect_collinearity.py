import numpy as np
import pandas as pd

from evalml.guardrails import detect_collinearity, detect_multicollinearity


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
    identity_matrix = np.identity(9)
    X = pd.DataFrame(data=identity_matrix)
    X[10] = X.iloc[:, 0] + X.iloc[:, 3] + X.iloc[:, 5]
    expected = set([0, 3, 5, 10])
    result = detect_multicollinearity(X)
    assert expected == set(result.keys())
