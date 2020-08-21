import numpy as np
import pandas as pd
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines import LightGBMClassifier
from evalml.problem_types import ProblemTypes

lgbm = importorskip('lightgbm', reason='Skipping test because lightgbm not installed')


def test_model_family():
    assert LightGBMClassifier.model_family == ModelFamily.LIGHTGBM


def test_problem_types():
    assert set(LightGBMClassifier.supported_problem_types) == {ProblemTypes.MULTICLASS, ProblemTypes.BINARY}


def test_et_parameters():
    clf = LightGBMClassifier(boosting_type="dart", learning_rate=0.5, n_estimators=20, max_depth=2, num_leaves=10, min_child_samples=10)
    expected_parameters = {
        "boosting_type": "dart",
        "learning_rate": 0.5,
        "n_estimators": 20,
        "max_depth": 2,
        "num_leaves": 10,
        "min_child_samples": 10,
        "n_jobs": -1
    }
    assert clf.parameters == expected_parameters


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = lgbm.sklearn.LGBMClassifier(random_state=0)
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

    sk_clf = lgbm.sklearn.LGBMClassifier(random_state=0)
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
    sk_clf = lgbm.sklearn.LGBMClassifier(random_state=0)
    sk_clf.fit(X, y)
    sk_feature_importance = sk_clf.feature_importances_

    clf.fit(X, y)
    feature_importance = clf.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)


def test_random_state(X_y_binary):
    X, y = X_y_binary

    clf = LightGBMClassifier(random_state=0)
    clf.fit(X, y)
    clf = LightGBMClassifier(random_state=np.random.RandomState(0))
    clf.fit(X, y)


def test_fit_string_features(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X['string_col'] = 'abc'

    clf = LightGBMClassifier()
    clf.fit(X, y)
