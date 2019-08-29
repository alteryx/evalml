from evalml import demos


def test_fraud():
    X, y = demos.load_fraud()
    assert X.shape == (99992, 12)


def test_wine():
    X, y = demos.load_wine()
    assert X.shape == (178, 13)


def test_breast_cancer():
    X, y = demos.load_breast_cancer()
    assert X.shape == (569, 30)


def test_diabetes():
    X, y = demos.load_diabetes()
    assert X.shape == (442, 10)
