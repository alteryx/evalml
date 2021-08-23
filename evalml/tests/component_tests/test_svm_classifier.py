import numpy as np
import pytest
from sklearn.svm import SVC

from evalml.model_family import ModelFamily
from evalml.pipelines import SVMClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert SVMClassifier.model_family == ModelFamily.SVM


def test_problem_types():
    assert set(SVMClassifier.supported_problem_types) == {
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    }


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    # we need to set probability to true in order to use `predict_proba`
    sk_svc = SVC(gamma="auto", probability=True, random_state=0)
    sk_svc.fit(X, y)
    y_pred_sk = sk_svc.predict(X)
    y_pred_proba_sk = sk_svc.predict_proba(X)

    svc = SVMClassifier()
    svc.fit(X, y)
    y_pred = svc.predict(X)
    y_pred_proba = svc.predict_proba(X)

    np.testing.assert_almost_equal(y_pred.values, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba.values, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_svc = SVC(gamma="auto", probability=True, random_state=0)
    sk_svc.fit(X, y)
    y_pred_sk = sk_svc.predict(X)
    y_pred_proba_sk = sk_svc.predict_proba(X)

    svc = SVMClassifier()
    svc.fit(X, y)

    y_pred = svc.predict(X)
    y_pred_proba = svc.predict_proba(X)

    np.testing.assert_almost_equal(y_pred.values, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba.values, y_pred_proba_sk, decimal=5)


@pytest.mark.parametrize("kernel", ["linear", "rbf", "sigmoid"])
def test_feature_importance(kernel, X_y_binary):
    X, y = X_y_binary

    svc = SVMClassifier(kernel=kernel)
    sk_svc = SVC(kernel=kernel, random_state=0)
    sk_svc.fit(X, y)

    if kernel == "linear":
        sk_feature_importance = sk_svc.coef_
    else:
        sk_feature_importance = np.zeros(sk_svc.n_features_in_)

    svc.fit(X, y)
    feature_importance = svc.feature_importance

    np.testing.assert_almost_equal(sk_feature_importance, feature_importance, decimal=5)
