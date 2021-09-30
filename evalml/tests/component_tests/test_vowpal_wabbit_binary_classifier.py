import warnings

import numpy as np

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.classifiers import (
    VowpalWabbitBinaryClassifier,
)
from evalml.problem_types import ProblemTypes
from vowpalwabbit.sklearn_vw import VWClassifier



def test_model_family():
    assert VowpalWabbitBinaryClassifier.model_family == ModelFamily.VOWPAL_WABBIT


def test_problem_types():
    assert set(VowpalWabbitBinaryClassifier.supported_problem_types) == {
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    }


def test_vw_parameters():
    vw = VowpalWabbitBinaryClassifier()
    expected_parameters = {
        "loss_function": "logistic",
        "learning_rate": 0.5,
        "decay_learning_rate": 0.95,
        "power_t": 1.0,
    }
    assert vw.parameters == expected_parameters

    vw = VowpalWabbitBinaryClassifier(
        loss_function="classic", learning_rate=0.1, decay_learning_rate=1.0, power_t=0.1
    )
    expected_parameters = {
        "loss_function": "classic",
        "learning_rate": 0.1,
        "decay_learning_rate": 1.0,
        "power_t": 0.1,
    }
    assert vw.parameters == expected_parameters


def test_fit_predict(X_y_binary):
    X, y = X_y_binary
    vw = VowpalWabbitBinaryClassifier()

    vw.fit(X, y)
    y_pred_sk = vw.predict(X)
    y_pred_proba_sk = vw.predict_proba(X)

    clf = VWClassifier()
    clf.fit(X, y)
    y_pred = clf.predict(X)
    y_pred_proba = clf.predict_proba(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)
    np.testing.assert_almost_equal(y_pred_proba_sk, y_pred_proba, decimal=5)



def test_feature_importance(X_y_binary):
    X, y = X_y_binary

    vw = VowpalWabbitBinaryClassifier()

    vw.fit(X, y)

    clf = VWClassifier()
    clf.fit(X, y)
    np.testing.assert_almost_equal(
        clf.get_coefs()[0], vw.feature_importance, decimal=5
    )

