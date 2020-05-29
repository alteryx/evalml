import numpy as np
from sklearn.linear_model import SGDClassifier as SKElasticNetClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import (
    ElasticNetClassifier
)
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ElasticNetClassifier.model_family == ModelFamily.LINEAR_MODEL


def test_en_parameters():

    clf = ElasticNetClassifier(alpha=0.75, l1_ratio=0.5, random_state=2)
    expected_parameters = {
        "alpha": 0.75,
        "l1_ratio": 0.5
    }

    assert clf.parameters == expected_parameters


def test_problem_types():
    assert ProblemTypes.BINARY in ElasticNetClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in ElasticNetClassifier.supported_problem_types
    assert len(ElasticNetClassifier.supported_problem_types) == 2


def test_fit_predict_binary(X_y):
    X, y = X_y

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=-1,
                                    random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ElasticNetClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_fit_predict_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=-1,
                                    random_state=0)
    sk_clf.fit(X, y)
    y_pred_sk = sk_clf.predict(X)
    y_pred_proba_sk = sk_clf.predict_proba(X)

    clf = ElasticNetClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred, y_pred_sk, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba, y_pred_proba_sk, decimal=5)


def test_feature_importances(X_y):
    X, y = X_y

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=-1,
                                    random_state=0)
    sk_clf.fit(X, y)

    clf = ElasticNetClassifier()
    clf.fit(X, y)

    np.testing.assert_almost_equal(sk_clf.coef_[0], clf.feature_importances, decimal=5)
