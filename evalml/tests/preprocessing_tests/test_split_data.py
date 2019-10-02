import pandas as pd

from evalml.preprocessing import split_data


def test_split_regression(X_y_reg):
    X, y = X_y_reg
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25


def test_split_classification(X_y):
    X, y = X_y
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.25)
    assert len(X_train) == 75
    assert len(X_test) == 25
    assert len(y_train) == 75
    assert len(y_test) == 25
