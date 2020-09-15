
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingEstimatorsError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    BaselineClassifier,
    RandomForestClassifier
)
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import _nonstackable_model_families


@pytest.fixture
def stackable_classifiers(all_classification_estimator_classes):
    estimators = [estimator_class() for estimator_class in all_classification_estimator_classes
                  if estimator_class.model_family not in _nonstackable_model_families and
                  estimator_class.model_family != ModelFamily.ENSEMBLE]
    return estimators


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_without_estimators_kwarg(stackable_classifiers):
    with pytest.raises(EnsembleMissingEstimatorsError, match='must be passed to the constructor as a keyword argument'):
        StackedEnsembleClassifier()


def test_stacked_ensemble_nonstackable_model_families(all_classification_estimator_classes):
    with pytest.raises(ValueError, match="Classifiers with any of the following model families cannot be used as base estimators in StackedEnsembleClassifier"):
        StackedEnsembleClassifier(estimators=[BaselineClassifier()])


def test_stacked_ensemble_init_with_multiple_same_estimators(stackable_classifiers, X_y_binary):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    estimators = [RandomForestClassifier(), RandomForestClassifier()]
    clf = StackedEnsembleClassifier(estimators=estimators)
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
    base = StackedEnsembleClassifier(estimators=[RandomForestClassifier(), RandomForestClassifier()])
    clf = StackedEnsembleClassifier(estimators=[RandomForestClassifier(), RandomForestClassifier()], final_estimator=base)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_stacked_ensemble_parameters(stackable_classifiers):
    clf = StackedEnsembleClassifier(estimators=stackable_classifiers)
    expected_parameters = {
        "estimators": stackable_classifiers,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters


def test_stacked_problem_types():
    assert ProblemTypes.BINARY in StackedEnsembleClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in StackedEnsembleClassifier.supported_problem_types
    assert len(StackedEnsembleClassifier.supported_problem_types) == 2


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_stacked_fit_predict(X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        num_classes = 2
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        num_classes = 3

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()
    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba).all().all()

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=RandomForestClassifier())
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

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented for StackedEnsembleClassifier"):
        clf.feature_importance
