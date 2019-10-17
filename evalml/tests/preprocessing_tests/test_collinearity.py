import pandas as pd

from evalml.preprocessing import detect_multicollinearity


def test_detect_multicollinearity():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    result = detect_multicollinearity(X)
    assert set(["a", "b", "c", "d"]) == set(result.keys())
