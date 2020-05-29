import numpy as np
from sklearn.linear_model import ElasticNet as SKElasticNetRegressor

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.regressors import (
    ElasticNetRegressor
)
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ElasticNetRegressor.model_family == ModelFamily.LINEAR_MODEL


def test_problem_types():
    assert ProblemTypes.REGRESSION in ElasticNetRegressor.supported_problem_types
    assert len(ElasticNetRegressor.supported_problem_types) == 1


def test_fit_predict(X_y):
    X, y = X_y

    sk_clf = SKElasticNetRegressor(alpha=0.5,
                                   l1_ratio=0.5,
                                   random_state=0,
                                   normalize=False,
                                   max_iter=1000)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)

    clf = ElasticNetRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)


def test_feature_importances(X_y):
    X, y = X_y

    sk_clf = SKElasticNetRegressor(alpha=0.5,
                                   l1_ratio=0.5,
                                   random_state=0,
                                   normalize=False,
                                   max_iter=1000)
    sk_clf.fit(X, y)

    clf = ElasticNetRegressor()
    clf.fit(X, y)

    np.testing.assert_almost_equal(sk_clf.coef_, clf.feature_importances, decimal=5)
