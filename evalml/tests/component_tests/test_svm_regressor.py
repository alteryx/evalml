import numpy as np
import pytest
from sklearn.svm import SVR

from evalml.model_family import ModelFamily
from evalml.pipelines import SVMRegressor
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert SVMRegressor.model_family == ModelFamily.SVM


def test_problem_types():
    assert set(SVMRegressor.supported_problem_types) == {ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION}


def test_fit_predict_regression(X_y_regression):
    X, y = X_y_regression

    sk_svr = SVR()
    sk_svr.fit(X, y)
    y_pred_sk = sk_svr.predict(X)

    svr = SVMRegressor()
    svr.fit(X, y)
    y_pred = svr.predict(X)

    np.testing.assert_almost_equal(y_pred.to_series().values, y_pred_sk, decimal=5)


@pytest.mark.parametrize('kernel', ['linear', 'rbf', 'sigmoid'])
def test_feature_importance(kernel, X_y_regression):
    X, y = X_y_regression

    svr = SVMRegressor(kernel=kernel)
    sk_svr = SVR(kernel=kernel)
    sk_svr.fit(X, y)

    if kernel == 'linear':
        sk_feature_importance = sk_svr.coef_
    else:
        sk_feature_importance = np.zeros(sk_svr.n_features_in_)

    svr.fit(X, y)
    feature_importance = svr.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
