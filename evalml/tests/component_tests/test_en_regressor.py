import numpy as np
import pytest
from sklearn.linear_model import ElasticNet as SKElasticNetRegressor

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.regressors import (
    ElasticNetRegressor
)
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ElasticNetRegressor.model_family == ModelFamily.LINEAR_MODEL


def test_en_parameters():

    clf = ElasticNetRegressor(alpha=0.75, l1_ratio=0.5, random_state=2)
    expected_parameters = {
        "alpha": 0.75,
        "l1_ratio": 0.5
    }

    assert clf.parameters == expected_parameters


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


def test_clone(X_y):
    X, y = X_y
    clf = ElasticNetRegressor(normalize=True)
    clf.fit(X, y)
    predicted = clf.predict(X)
    assert isinstance(predicted, type(np.array([])))

    clf_clone = clf.clone()
    with pytest.raises(MethodPropertyNotFoundError):
        clf_clone.predict(X)

    assert clf_clone._component_obj.normalize is True

    clf_clone.fit(X, y)
    predicted_clone = clf_clone.predict(X)
    np.testing.assert_almost_equal(predicted, predicted_clone)
