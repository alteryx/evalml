import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import ElasticNetClassifier
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ElasticNetClassifier.model_family == ModelFamily.LINEAR_MODEL


def test_problem_types():
    assert set(ElasticNetClassifier.supported_problem_types) == {
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    }


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

    sk_clf = LogisticRegression(
        C=1.0,
        penalty="elasticnet",
        l1_ratio=0.15,
        n_jobs=-1,
        multi_class="auto",
        solver="saga",
        random_state=0,
    )
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ElasticNetClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.values, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = LogisticRegression(
        C=1.0,
        penalty="elasticnet",
        l1_ratio=0.15,
        n_jobs=-1,
        multi_class="auto",
        solver="saga",
        random_state=0,
    )
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ElasticNetClassifier()
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ElasticNetClassifier)

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred.values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.values, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    sk_clf = LogisticRegression(
        C=1.0,
        penalty="elasticnet",
        l1_ratio=0.15,
        n_jobs=1,
        multi_class="auto",
        solver="saga",
        random_state=0,
    )
    sk_clf.fit(X, y)

    clf = ElasticNetClassifier(n_jobs=1)
    clf.fit(X, y)

    np.testing.assert_almost_equal(
        sk_clf.coef_.flatten(),
        clf.feature_importance,
        decimal=5,
    )


def test_feature_importance_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = LogisticRegression(
        C=1.0,
        penalty="elasticnet",
        l1_ratio=0.15,
        n_jobs=1,
        multi_class="auto",
        solver="saga",
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as w1:
        sk_clf.fit(X, y)
    assert len(w1) > 0

    with warnings.catch_warnings(record=True) as w2:
        clf = ElasticNetClassifier(n_jobs=1)
        clf.fit(X, y)
    assert len(w2) == 0

    sk_features = np.linalg.norm(sk_clf.coef_, axis=0, ord=2)

    np.testing.assert_almost_equal(sk_features, clf.feature_importance, decimal=5)
