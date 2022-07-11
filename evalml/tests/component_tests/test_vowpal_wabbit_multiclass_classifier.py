import numpy as np
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import (
    VowpalWabbitMulticlassClassifier,
)
from evalml.problem_types import ProblemTypes


def test_vw_model_family():
    assert VowpalWabbitMulticlassClassifier.model_family == ModelFamily.VOWPAL_WABBIT


def test_vw_problem_types():
    assert set(VowpalWabbitMulticlassClassifier.supported_problem_types) == {
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    }


def test_vw_parameters():
    vw = VowpalWabbitMulticlassClassifier()
    expected_parameters = {
        "loss_function": "logistic",
        "learning_rate": 0.5,
        "decay_learning_rate": 1.0,
        "power_t": 0.5,
        "passes": 1,
    }
    assert vw.parameters == expected_parameters

    vw = VowpalWabbitMulticlassClassifier(
        loss_function="classic",
        learning_rate=0.1,
        decay_learning_rate=1.0,
        power_t=0.1,
        passes=2,
    )
    expected_parameters = {
        "loss_function": "classic",
        "learning_rate": 0.1,
        "decay_learning_rate": 1.0,
        "power_t": 0.1,
        "passes": 2,
    }
    assert vw.parameters == expected_parameters


def test_vw_fit_predict(X_y_multi, vw):

    X, y = X_y_multi
    vw_classifier = VowpalWabbitMulticlassClassifier()

    vw_classifier.fit(X, y)
    y_pred = vw_classifier.predict(X)
    y_pred_proba = vw_classifier.predict_proba(X)

    clf = vw.VWMultiClassifier(
        loss_function="logistic",
        learning_rate=0.5,
        decay_learning_rate=1.0,
        power_t=0.5,
    )
    clf.fit(X, y)
    y_pred_sk = clf.predict(X)
    y_pred_proba_sk = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba, decimal=5)


def test_vw_feature_importance(X_y_multi):
    X, y = X_y_multi
    vw = VowpalWabbitMulticlassClassifier()
    vw.fit(X, y)
    with pytest.raises(
        NotImplementedError,
        match="Feature importance is not implemented for the Vowpal Wabbit classifiers",
    ):
        vw.feature_importance
