import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier as SKElasticNetClassifier

from evalml.exceptions import MethodPropertyNotFoundError
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
        "l1_ratio": 0.5,
        "max_iter": 1000
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

    np.testing.assert_almost_equal(sk_clf.coef_.flatten(), clf.feature_importances, decimal=5)


def test_feature_importances_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=-1,
                                    random_state=0)
    sk_clf.fit(X, y)

    clf = ElasticNetClassifier()
    clf.fit(X, y)

    sk_features = np.linalg.norm(sk_clf.coef_, axis=0, ord=2)

    np.testing.assert_almost_equal(sk_features, clf.feature_importances, decimal=5)


def test_clone(X_y):
    X, y = X_y
    clf = ElasticNetClassifier(max_iter=500)
    clf.fit(X, y)
    predicted = clf.predict(X)
    assert isinstance(predicted, type(np.array([])))

    # Test unlearned clone
    clf_clone = clf.clone(learned=False)
    with pytest.raises(MethodPropertyNotFoundError):
        clf_clone.predict(X)
    assert clf_clone._component_obj.max_iter == 500

    clf_clone.fit(X, y)
    predicted_clone = clf_clone.predict(X)
    np.testing.assert_almost_equal(predicted, predicted_clone)

    # Test learned clone
    clf_clone = clf.clone()
    assert clf_clone._component_obj.max_iter == 500
    predicted_clone = clf_clone.predict(X)
    np.testing.assert_almost_equal(predicted, predicted_clone)
