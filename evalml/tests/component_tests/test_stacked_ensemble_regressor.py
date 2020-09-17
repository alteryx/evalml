
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    BaselineRegressor,
    Estimator,
    RandomForestRegressor
)
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleRegressor()
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleRegressor(input_pipelines=[])


def test_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(ValueError, match="Estimators with any of the following model families cannot be used as base estimators"):
        StackedEnsembleRegressor(input_pipelines=[BaselineRegressor()])


def test_stacked_ensemble_init_with_multiple_same_estimators(X_y_regression):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_regression
    estimators = [RandomForestRegressor(), RandomForestRegressor()]
    clf = StackedEnsembleRegressor(input_pipelines=estimators)
    expected_parameters = {
        "estimators": estimators,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_multilevel():
    # checks passing a stacked ensemble classifier as a final estimator
    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series(np.random.rand(50,))
    base = StackedEnsembleRegressor(input_pipelines=[RandomForestRegressor(), RandomForestRegressor()])
    clf = StackedEnsembleRegressor(input_pipelines=[RandomForestRegressor(), RandomForestRegressor()], final_estimator=base)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 1


def test_stacked_ensemble_final_estimator_without_component_obj(stackable_regressors):
    class MockRegressor(Estimator):
        name = "Mock Regressor"
        model_family = ModelFamily.RANDOM_FOREST
        supported_problem_types = [ProblemTypes.REGRESSION]

    with pytest.raises(ValueError, match='All estimators and final_estimator must have a valid ._component_obj'):
        StackedEnsembleRegressor(input_pipelines=stackable_regressors,
                                 final_estimator=MockRegressor())
    with pytest.raises(ValueError, match='All estimators and final_estimator must have a valid ._component_obj'):
        StackedEnsembleRegressor(input_pipelines=[MockRegressor()],
                                 final_estimator=RandomForestRegressor())


def test_stacked_fit_predict_regression(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(input_pipelines=stackable_regressors)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()

    clf = StackedEnsembleRegressor(input_pipelines=stackable_regressors, final_estimator=RandomForestRegressor())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


@patch('evalml.pipelines.components.ensemble.StackedEnsembleRegressor.fit')
def test_stacked_feature_importance(mock_fit, X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(input_pipelines=stackable_regressors)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented"):
        clf.feature_importance
