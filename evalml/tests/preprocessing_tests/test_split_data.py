from evalml.preprocessing import split_data


def test_split_regression(X_y_regression):
    X, y = X_y_regression
    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct, regression=True)
    test_size = len(X) * test_pct
    train_size = len(X) - test_size
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size


def test_split_classification(X_y_binary):
    X, y = X_y_binary
    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct)
    test_size = len(X) * 0.25
    train_size = len(X) - test_size
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size
