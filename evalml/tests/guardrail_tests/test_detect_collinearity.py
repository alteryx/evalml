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
    col = np.random.randn(100, 10)
    noise = np.random.randn(100)
    col[:, 9] = 3 * col[:, 2] + 1.5 * col[:, 3] + 5 * col[:, 6] + .5 * noise
    X = pd.DataFrame(data=col)
    expected = set([2, 3, 6, 9])
    result = detect_multicollinearity(X)
    assert expected == set(result.keys())
