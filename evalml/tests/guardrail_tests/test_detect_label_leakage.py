import pandas as pd

from evalml.guardrails import (
    detect_label_leakage,
    detect_numerical_categorical_correlation
)


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


def test_detect_num_cat():
    # category has different means
    num = pd.Series([10, 9, 11, 10, 11, 51, 49, 50, 52, 51], name='num')
    cat = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'], name='cat')

    stat, pvalue = detect_numerical_categorical_correlation(num, cat)
    assert pvalue < 0.05

    # category has exact mean
    num = pd.Series([10, 9, 11, 10, 11, 10, 9, 11, 10, 11], name='num')
    cat = pd.Series(['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B'], name='cat')

    stat, pvalue = detect_numerical_categorical_correlation(num, cat)
    assert pvalue == 1.0

    num = [10, 9, 11, 10, 11, 10, 9, 11, 10, 11]
    cat = ['A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B']

    stat, pvalue = detect_numerical_categorical_correlation(num, cat)
    assert pvalue == 1.0
