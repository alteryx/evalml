import urllib.request

import pandas as pd
import pytest

from evalml import demos


@pytest.fixture(autouse=True)
def set_testing_headers():
    opener = urllib.request.build_opener()
    opener.addheaders = [("Testing", "True")]
    urllib.request.install_opener(opener)


def test_fraud(fraud_local):
    X, y = demos.load_fraud()
    X_local, y_local = fraud_local
    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
    assert X.shape == (99992, 12)
    assert y.shape == (99992,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None


def test_wine(wine_local):
    X, y = demos.load_wine()
    X_local, y_local = wine_local
    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
    assert X.shape == (178, 13)
    assert y.shape == (178,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None


def test_breast_cancer(breast_cancer_local):
    X, y = demos.load_breast_cancer()
    X_local, y_local = breast_cancer_local
    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None


def test_diabetes(diabetes_local):
    X, y = demos.load_diabetes()
    X_local, y_local = diabetes_local
    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
    assert X.shape == (442, 10)
    assert y.shape == (442,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None


def test_churn(churn_local):
    X, y = demos.load_churn()
    X_local, y_local = churn_local
    pd.testing.assert_frame_equal(X, X_local)
    pd.testing.assert_series_equal(y, y_local)
    assert X.shape == (7043, 19)
    assert y.shape == (7043,)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.ww.schema is not None
    assert y.ww.schema is not None
