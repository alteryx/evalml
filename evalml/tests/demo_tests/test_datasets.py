import pandas as pd
import woodwork as ww

from evalml import demos


def test_fraud():
    X, y = demos.load_fraud()
    assert X.shape == (99992, 12)
    assert y.shape == (99992,)
    assert isinstance(X, ww.DataTable)
    assert isinstance(y, ww.DataColumn)

    X, y = demos.load_fraud(1000)
    assert X.shape == (1000, 12)
    assert y.shape == (1000,)

    X, y = demos.load_fraud(1000, return_pandas=True)
    assert X.shape == (1000, 12)
    assert y.shape == (1000,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_wine():
    X, y = demos.load_wine()
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert isinstance(X, ww.DataTable)
    assert isinstance(y, ww.DataColumn)

    X, y = demos.load_wine(return_pandas=True)
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_breast_cancer():
    X, y = demos.load_breast_cancer()
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert isinstance(X, ww.DataTable)
    assert isinstance(y, ww.DataColumn)

    X, y = demos.load_breast_cancer(return_pandas=True)
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_diabetes():
    X, y = demos.load_diabetes()
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    assert isinstance(X, ww.DataTable)
    assert isinstance(y, ww.DataColumn)

    X, y = demos.load_diabetes(return_pandas=True)
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_churn():
    X, y = demos.load_churn()
    assert X.shape == (7043, 19)
    assert y.shape == (7043,)
    assert isinstance(X, ww.DataTable)
    assert isinstance(y, ww.DataColumn)

    X, y = demos.load_churn(return_pandas=True)
    assert X.shape == (7043, 19)
    assert y.shape == (7043,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
