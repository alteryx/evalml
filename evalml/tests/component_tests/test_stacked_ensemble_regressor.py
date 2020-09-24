
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    BaselineRegressor,
    RandomForestClassifier,
    RandomForestRegressor
)
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.pipelines.utils import make_pipeline_from_components
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleRegressor()
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleRegressor(input_pipelines=[])


def test_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(ValueError, match="Pipelines with any of the following model families cannot be used as base pipelines"):
        StackedEnsembleRegressor(input_pipelines=[make_pipeline_from_components([BaselineRegressor()], ProblemTypes.REGRESSION)])


def test_stacked_different_input_pipelines_regression():
    input_pipelines = [make_pipeline_from_components([RandomForestRegressor()], ProblemTypes.REGRESSION),
                       make_pipeline_from_components([RandomForestClassifier()], ProblemTypes.BINARY)]
    with pytest.raises(ValueError, match="All pipelines must have the same problem type."):
        StackedEnsembleRegressor(input_pipelines=input_pipelines)


def test_stacked_ensemble_init_with_multiple_same_estimators(X_y_regression, linear_regression_pipeline_class):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_regression
    input_pipelines = [linear_regression_pipeline_class(parameters={}),
                       linear_regression_pipeline_class(parameters={})]
    clf = StackedEnsembleRegressor(input_pipelines=input_pipelines)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        'cv': None,
        'n_jobs': 1
    }
    assert clf.parameters == expected_parameters
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_multilevel(linear_regression_pipeline_class):
    # checks passing a stacked ensemble classifier as a final estimator
    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series(np.random.rand(50,))
    base = StackedEnsembleRegressor(input_pipelines=[linear_regression_pipeline_class(parameters={})])
    clf = StackedEnsembleRegressor(input_pipelines=[linear_regression_pipeline_class(parameters={})],
                                   final_estimator=base)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 1


def test_stacked_fit_predict_regression(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    input_pipelines = [make_pipeline_from_components([regressor], ProblemTypes.REGRESSION)
                       for regressor in stackable_regressors]
    clf = StackedEnsembleRegressor(input_pipelines=input_pipelines)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    clf = StackedEnsembleRegressor(input_pipelines=input_pipelines, final_estimator=RandomForestRegressor())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()


@patch('evalml.pipelines.components.ensemble.StackedEnsembleRegressor.fit')
def test_stacked_feature_importance(mock_fit, X_y_regression, stackable_regressors):
    X, y = X_y_regression
    input_pipelines = [make_pipeline_from_components([regressor], ProblemTypes.REGRESSION)
                       for regressor in stackable_regressors]
    clf = StackedEnsembleRegressor(input_pipelines=input_pipelines)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented"):
        clf.feature_importance
