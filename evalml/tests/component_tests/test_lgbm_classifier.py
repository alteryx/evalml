import numpy as np
from lightgbm.sklearn import LGBMClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines import LightGBMClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert LightGBMClassifier.model_family == ModelFamily.LIGHTGBM


def test_problem_types():
    assert ProblemTypes.BINARY in LightGBMClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in LightGBMClassifier.supported_problem_types
    assert len(LightGBMClassifier.supported_problem_types) == 2


def test_et_parameters():
    clf = LightGBMClassifier(boosting_type="dart", learning_rate=0.5, n_estimators=20)
    expected_parameters = {
        "boosting_type": "dart",
        "learning_rate": 0.5,
        "n_estimators": 20,
        "n_jobs": -1
    }
    assert clf.parameters == expected_parameters


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = LGBMClassifier(random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = LightGBMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = LGBMClassifier(random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = LightGBMClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    clf = LightGBMClassifier()
    sk_clf = LGBMClassifier(random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
