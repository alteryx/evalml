import pandas as pd

from evalml.guardrails import detect_label_leakage


def test_detect_label_leakage():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]

    y = y.astype(bool)

    result = detect_label_leakage(X, y)

    assert set(["a", "b", "c", "d"]) == set(result.keys())

def test_invalid_target_dtype():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]

    y = y.astype('O')

    result = detect_label_leakage(X, y)
    assert set() == set(result.keys())

