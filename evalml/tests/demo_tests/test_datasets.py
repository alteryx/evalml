from evalml import demos


def test_fraud():
    X, y = demos.load_fraud()
    assert X.shape == (99992, 12)
    assert y.shape == (99992,)

    X, y = demos.load_fraud(1000)
    assert X.shape == (1000, 12)
    assert y.shape == (1000,)


def test_wine():
    X, y = demos.load_wine()
    assert X.shape == (178, 13)
    assert y.shape == (178,)


def test_breast_cancer():
    X, y = demos.load_breast_cancer()
    assert X.shape == (569, 30)
    assert y.shape == (569,)


def test_diabetes():
    X, y = demos.load_diabetes()
    assert X.shape == (442, 10)
    assert y.shape == (442,)


def test_churn():
    X, y = demos.load_churn()
    assert X.shape == (7043, 19)
    assert y.shape == (7043,)
