import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor as SKExtraTreesRegressor

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines import ExtraTreesRegressor
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ExtraTreesRegressor.model_family == ModelFamily.EXTRA_TREES


def test_problem_types():
    assert ProblemTypes.REGRESSION in ExtraTreesRegressor.supported_problem_types
    assert len(ExtraTreesRegressor.supported_problem_types) == 1


def test_et_parameters():
    clf = ExtraTreesRegressor(n_estimators=20, max_features="auto", max_depth=5, random_state=2)
    expected_parameters = {
        "n_estimators": 20,
        "max_features": "auto",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "n_jobs": -1
    }
    assert clf.parameters == expected_parameters


def test_fit_predict(X_y):
    X, y = X_y

    sk_clf = SKExtraTreesRegressor(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)

    clf = ExtraTreesRegressor()
    clf.fit(X, y)
    y_pred = clf.predict(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)


def test_feature_importance(X_y):
    X, y = X_y

    # testing that feature importance can't be called before fit
    clf = ExtraTreesRegressor()
    with pytest.raises(MethodPropertyNotFoundError):
        feature_importance = clf.feature_importance

    sk_clf = SKExtraTreesRegressor(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
