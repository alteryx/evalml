import numpy as np
import pytest

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators.regressors import VowpalWabbitRegressor
from evalml.problem_types import ProblemTypes


def test_vw_model_family():
    assert VowpalWabbitRegressor.model_family == ModelFamily.VOWPAL_WABBIT


def test_vw_problem_types():
    assert set(VowpalWabbitRegressor.supported_problem_types) == {
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    }


def test_vw_parameters():
    vw = VowpalWabbitRegressor()
    expected_parameters = {
        "learning_rate": 0.5,
        "decay_learning_rate": 1.0,
        "power_t": 0.5,
        "passes": 1,
    }
    assert vw.parameters == expected_parameters

    vw = VowpalWabbitRegressor(
        learning_rate=0.1,
        decay_learning_rate=1.0,
        power_t=0.1,
        passes=2,
    )
    expected_parameters = {
        "learning_rate": 0.1,
        "decay_learning_rate": 1.0,
        "power_t": 0.1,
        "passes": 2,
    }
    assert vw.parameters == expected_parameters


def test_vw_fit_predict(X_y_regression, vw):

    X, y = X_y_regression
    vw_regressor = VowpalWabbitRegressor()

    vw_regressor.fit(X, y)
    y_pred = vw_regressor.predict(X)

    clf = vw.VWRegressor(
        learning_rate=0.5,
        decay_learning_rate=1.0,
        power_t=0.5,
        passes=1,
    )
    clf.fit(X, y)
    y_pred_sk = clf.predict(X)

    np.testing.assert_almost_equal(y_pred_sk, y_pred, decimal=5)


def test_vw_feature_importance(X_y_regression):
    X, y = X_y_regression
    vw = VowpalWabbitRegressor()
    vw.fit(X, y)
    with pytest.raises(
        NotImplementedError,
        match="Feature importance is not implemented for the Vowpal Wabbit regressor",
    ):
        vw.feature_importance
