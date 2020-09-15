
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingEstimatorsError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    BaselineRegressor,
    RandomForestRegressor
)
from evalml.pipelines.components.ensemble import StackedEnsembleRegressor
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import _nonstackable_model_families


@pytest.fixture
def stackable_regressors(all_regression_estimators_classes):
    estimators = [estimator_class() for estimator_class in all_regression_estimators_classes
                  if estimator_class.model_family not in _nonstackable_model_families and
                  estimator_class.model_family != ModelFamily.ENSEMBLE]
    return estimators


def test_stacked_model_family():
    assert StackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_without_estimators_kwarg(stackable_regressors):
    with pytest.raises(EnsembleMissingEstimatorsError, match='must be passed to the constructor as a keyword argument'):
        StackedEnsembleRegressor()


def test_stacked_ensemble_nonstackable_model_families(all_regression_estimators_classes):
    with pytest.raises(ValueError, match="Regressors with any of the following model families cannot be used as base estimators in StackedEnsembleRegressor"):
        StackedEnsembleRegressor(estimators=[BaselineRegressor()])


def test_stacked_ensemble_init_with_multiple_same_estimators(X_y_regression):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_regression
    estimators = [RandomForestRegressor(), RandomForestRegressor()]
    clf = StackedEnsembleRegressor(estimators=estimators)
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
    base = StackedEnsembleRegressor(estimators=[RandomForestRegressor(), RandomForestRegressor()])
    clf = StackedEnsembleRegressor(estimators=[RandomForestRegressor(), RandomForestRegressor()], final_estimator=base)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_parameters(stackable_regressors):
    clf = StackedEnsembleRegressor(estimators=stackable_regressors, final_estimator=None, random_state=2)
    expected_parameters = {
        "estimators": stackable_regressors,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters


def test_stacked_problem_types():
    assert ProblemTypes.REGRESSION in StackedEnsembleRegressor.supported_problem_types
    assert len(StackedEnsembleRegressor.supported_problem_types) == 1


def test_stacked_fit_predict_regression(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(estimators=stackable_regressors)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()

    clf = StackedEnsembleRegressor(estimators=stackable_regressors, final_estimator=RandomForestRegressor())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


@patch('evalml.pipelines.components.ensemble.StackedEnsembleRegressor.fit')
def test_stacked_feature_importance(mock_fit, X_y_regression, stackable_regressors):
    X, y = X_y_regression
    clf = StackedEnsembleRegressor(estimators=stackable_regressors)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented for StackedEnsembleRegressor"):
        clf.feature_importance
