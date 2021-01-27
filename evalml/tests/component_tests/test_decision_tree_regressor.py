import numpy as np
from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor

from evalml.model_family import ModelFamily
from evalml.pipelines import DecisionTreeRegressor
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert DecisionTreeRegressor.model_family == ModelFamily.DECISION_TREE


def test_problem_types():
    assert set(DecisionTreeRegressor.supported_problem_types) == {ProblemTypes.REGRESSION,
                                                                  ProblemTypes.TIME_SERIES_REGRESSION}


def test_fit_predict(X_y_regression):
    X, y = X_y_regression

    sk_clf = SKDecisionTreeRegressor(max_depth=6, max_features='auto', random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)

    clf = DecisionTreeRegressor()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, DecisionTreeRegressor)

    y_pred = clf.predict(X)
    np.testing.assert_almost_equal(y_pred_sk, y_pred.to_series().values, decimal=5)


def test_feature_importance(X_y_regression):
    X, y = X_y_regression

    clf = DecisionTreeRegressor()
    sk_clf = SKDecisionTreeRegressor(max_depth=6, max_features='auto', random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
