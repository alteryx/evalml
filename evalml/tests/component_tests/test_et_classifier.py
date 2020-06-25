import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesClassifier as SKExtraTreesClassifier

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines import ExtraTreesClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ExtraTreesClassifier.model_family == ModelFamily.EXTRA_TREES


def test_problem_types():
    assert ProblemTypes.BINARY in ExtraTreesClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in ExtraTreesClassifier.supported_problem_types
    assert len(ExtraTreesClassifier.supported_problem_types) == 2


def test_et_parameters():
    clf = ExtraTreesClassifier(n_estimators=20, max_features="auto", max_depth=5, random_state=2)
    expected_parameters = {
        "n_estimators": 20,
        "max_features": "auto",
        "max_depth": 5,
        "min_samples_split": 2,
        "min_weight_fraction_leaf": 0.0,
        "n_jobs": -1
    }
    assert clf.parameters == expected_parameters


def test_fit_predict_binary(X_y):
    X, y = X_y

    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ExtraTreesClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_feature_importance(X_y):
    X, y = X_y

    # testing that feature importances can't be called before fit
    clf = ExtraTreesClassifier()
    with pytest.raises(MethodPropertyNotFoundError):
        feature_importance = clf.feature_importance

    sk_clf = SKExtraTreesClassifier(max_depth=6, random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importance_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
