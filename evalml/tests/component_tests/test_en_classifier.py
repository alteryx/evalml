import numpy as np
import pytest
from sklearn.linear_model import SGDClassifier as SKElasticNetClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import (
    ElasticNetClassifier
)
from evalml.problem_types import ProblemTypes


def test_model_family():
    assert ElasticNetClassifier.model_family == ModelFamily.LINEAR_MODEL


def test_problem_types():
    assert set(ElasticNetClassifier.supported_problem_types) == {ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                                                 ProblemTypes.TIME_SERIES_BINARY,
                                                                 ProblemTypes.TIME_SERIES_MULTICLASS}


def test_fit_predict_binary(X_y_binary):
    X, y = X_y_binary

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

    np.testing.assert_almost_equal(y_pred_sk, y_pred.to_series().values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.to_dataframe().values, decimal=5)


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
    fitted = clf.fit(X, y)
    assert isinstance(fitted, ElasticNetClassifier)

    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred.to_series().values, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba.to_dataframe().values, decimal=5)


def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=1,
                                    random_state=0)
    sk_clf.fit(X, y)

    clf = ElasticNetClassifier(n_jobs=1)
    clf.fit(X, y)

    np.testing.assert_almost_equal(sk_clf.coef_.flatten(), clf.feature_importance, decimal=5)


def test_feature_importance_multi(X_y_multi):
    X, y = X_y_multi

    sk_clf = SKElasticNetClassifier(loss="log",
                                    penalty="elasticnet",
                                    alpha=0.5,
                                    l1_ratio=0.5,
                                    n_jobs=1,
                                    random_state=0)
    sk_clf.fit(X, y)

    clf = ElasticNetClassifier(n_jobs=1)
    clf.fit(X, y)

    sk_features = np.linalg.norm(sk_clf.coef_, axis=0, ord=2)

    np.testing.assert_almost_equal(sk_features, clf.feature_importance, decimal=5)


def test_overwrite_loss_parameter_in_kwargs():

    with pytest.warns(expected_warning=UserWarning) as warnings:
        en = ElasticNetClassifier(loss="hinge")

    assert len(warnings) == 1
    # check that the message matches
    assert warnings[0].message.args[0] == ("Parameter loss is being set to 'log' so that ElasticNetClassifier can predict probabilities"
                                           ". Originally received 'hinge'.")

    assert en.parameters['loss'] == 'log'
