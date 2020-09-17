
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    BaselineClassifier,
    Estimator,
    RandomForestClassifier
)
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleClassifier()
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleClassifier(input_pipelines=[])


def test_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(ValueError, match="Estimators with any of the following model families cannot be used as base estimators"):
        StackedEnsembleClassifier(input_pipelines=[BaselineClassifier()])


def test_stacked_ensemble_init_with_multiple_same_estimators(X_y_binary):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    estimators = [RandomForestClassifier(), RandomForestClassifier()]
    clf = StackedEnsembleClassifier(input_pipelines=estimators)
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
    y = pd.Series([1, 0] * 25)
    base = StackedEnsembleClassifier(input_pipelines=[RandomForestClassifier(), RandomForestClassifier()])
    clf = StackedEnsembleClassifier(input_pipelines=[RandomForestClassifier(), RandomForestClassifier()], final_estimator=base)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_problem_types():
    assert ProblemTypes.BINARY in StackedEnsembleClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in StackedEnsembleClassifier.supported_problem_types
    assert StackedEnsembleClassifier.supported_problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]


def test_stacked_ensemble_final_estimator_without_component_obj(stackable_classifiers):
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.RANDOM_FOREST
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    with pytest.raises(ValueError, match='All estimators and final_estimator must have a valid ._component_obj'):
        StackedEnsembleClassifier(input_pipelines=stackable_classifiers,
                                  final_estimator=MockEstimator())
    with pytest.raises(ValueError, match='All estimators and final_estimator must have a valid ._component_obj'):
        StackedEnsembleClassifier(input_pipelines=[MockEstimator()],
                                  final_estimator=RandomForestClassifier())


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_stacked_fit_predict(X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        num_classes = 2
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        num_classes = 3

    clf = StackedEnsembleClassifier(input_pipelines=stackable_classifiers)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()
    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba).all().all()

    clf = StackedEnsembleClassifier(input_pipelines=stackable_classifiers, final_estimator=RandomForestClassifier())
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()
    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba).all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@patch('evalml.pipelines.components.ensemble.StackedEnsembleClassifier.fit')
def test_stacked_feature_importance(mock_fit, X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi

    clf = StackedEnsembleClassifier(input_pipelines=stackable_classifiers)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented"):
        clf.feature_importance
