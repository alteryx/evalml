import numpy as np
from sklearn.neighbors import KNeighborsClassifier as SKKNeighborsClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines import KNeighborsClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert KNeighborsClassifier.model_family == ModelFamily.K_NEIGHBORS


def test_problem_types():
    assert set(KNeighborsClassifier.supported_problem_types) == {
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    }


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = SKKNeighborsClassifier()
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = KNeighborsClassifier()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, KNeighborsClassifier)

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKKNeighborsClassifier()
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = KNeighborsClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    clf = KNeighborsClassifier()
    clf.fit(X, y)
    np.testing.assert_equal(clf.feature_importance, np.zeros(X.shape[1]))
