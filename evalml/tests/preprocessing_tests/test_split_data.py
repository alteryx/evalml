from evalml.preprocessing import split_data


def test_split_regression(X_y_regression):
    X, y = X_y_regression
    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct, regression=True)
    test_size = X.shape[0] * test_pct
    train_size = X.shape[0] - test_size
    assert X_train.shape[0] == train_size
    assert X_test.shape[0] == test_size
    assert y_train.shape[0] == train_size
    assert y_test.shape[0] == test_size


def test_split_classification(X_y_binary):
    X, y = X_y_binary
    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct)
    test_size = X.shape[0] * 0.25
    train_size = X.shape[0] - test_size
    assert X_train.shape[0] == train_size
    assert X_test.shape[0] == test_size
    assert y_train.shape[0] == train_size
    assert y_test.shape[0] == test_size
