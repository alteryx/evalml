import numpy as np
import pytest
from pytest import importorskip

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import (
    VowpalWabbitMulticlassClassifier,
)
from evalml.problem_types import ProblemTypes

vw = importorskip(
    "vowpalwabbit.sklearn_vw",
    reason="Skipping test because vowpal wabbit not installed",
)


def test_model_family():
    assert VowpalWabbitMulticlassClassifier.model_family == ModelFamily.VOWPAL_WABBIT


def test_problem_types():
    assert set(VowpalWabbitMulticlassClassifier.supported_problem_types) == {
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    }


def test_vw_parameters():
    vw = VowpalWabbitMulticlassClassifier()
    expected_parameters = {
        "loss_function": "logistic",
        "learning_rate": 0.5,
        "decay_learning_rate": 0.95,
        "power_t": 1.0,
    }
    assert vw.parameters == expected_parameters

    vw = VowpalWabbitMulticlassClassifier(
        loss_function="classic", learning_rate=0.1, decay_learning_rate=1.0, power_t=0.1
    )
    expected_parameters = {
        "loss_function": "classic",
        "learning_rate": 0.1,
        "decay_learning_rate": 1.0,
        "power_t": 0.1,
    }
    assert vw.parameters == expected_parameters


def test_fit_predict(X_y_multi):
    X, y = X_y_multi
    vw_classifier = VowpalWabbitMulticlassClassifier()

    vw_classifier.fit(X, y)
    y_pred_sk = vw_classifier.predict(X)
    y_pred_proba_sk = vw_classifier.predict_proba(X)

    clf = vw.VWMultiClassifier(
        loss_function="logistic",
        learning_rate=0.5,
        decay_learning_rate=0.95,
        power_t=1.0,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba, decimal=5)


def test_feature_importance(X_y_multi):
    X, y = X_y_multi
    vw = VowpalWabbitMulticlassClassifier()
    vw.fit(X, y)
    with pytest.raises(
        NotImplementedError,
        match="Feature importance is not implemented for the Vowpal Wabbit classifiers",
    ):
        vw.feature_importance
