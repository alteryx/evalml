
import numpy as np
import pytest

from evalml.exceptions import EnsembleMissingEstimatorsError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import RandomForestClassifier
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.problem_types import ProblemTypes


@pytest.fixture
def stackable_classifiers(all_classification_estimator_classes):
    estimators = [estimator_class() for estimator_class in all_classification_estimator_classes if estimator_class.model_family != ModelFamily.ENSEMBLE and estimator_class.model_family != ModelFamily.BASELINE]
    return estimators


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_ensemble_init_without_estimators_kwarg(stackable_classifiers):
    with pytest.raises(EnsembleMissingEstimatorsError):
        StackedEnsembleClassifier()


def test_stacked_ensemble_init_with_multiple_same_estimators(stackable_classifiers):
    # Checks that it is okay to pass multiple of the same type of estimator
    clf = StackedEnsembleClassifier(estimators=stackable_classifiers + stackable_classifiers, final_estimator=None, random_state=2)
    expected_parameters = {
        "estimators": stackable_classifiers + stackable_classifiers,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters


def test_stacked_ensemble_parameters(stackable_classifiers):
    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=None, random_state=2)
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

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=None, random_state=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()
    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred).all().all()

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=RandomForestClassifier(), random_state=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()
    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred).all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_stacked_feature_importance_rf(X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=None, random_state=2)
    clf.fit(X, y)
    assert not np.isnan(clf.feature_importance).all().all()

    clf = StackedEnsembleClassifier(estimators=stackable_classifiers, final_estimator=RandomForestClassifier(), random_state=2)
    clf.fit(X, y)
    assert not np.isnan(clf.feature_importance).all().all()
