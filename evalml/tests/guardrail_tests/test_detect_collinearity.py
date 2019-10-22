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
    result = detect_multicollinearity(X)
    assert set(["a", "b", "c", "d"]) == set(result.keys())
