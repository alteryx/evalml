from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline
)
from evalml.pipelines.components import (
    BaselineClassifier,
    RandomForestClassifier
)
from evalml.pipelines.components.ensemble import StackedEnsembleClassifier
from evalml.problem_types import ProblemTypes


def test_stacked_model_family():
    assert StackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_stacked_default_parameters():
    assert StackedEnsembleClassifier.default_parameters == {'final_estimator': None,
                                                            'cv': None,
                                                            'n_jobs': -1
                                                            }


def test_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleClassifier()
    with pytest.raises(EnsembleMissingPipelinesError, match='must not be None or an empty list.'):
        StackedEnsembleClassifier(input_pipelines=[])


def test_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(ValueError, match="Pipelines with any of the following model families cannot be used as base pipelines"):
        StackedEnsembleClassifier(input_pipelines=[BinaryClassificationPipeline([BaselineClassifier])])


def test_stacked_different_input_pipelines_classification():
    input_pipelines = [BinaryClassificationPipeline([RandomForestClassifier]),
                       MulticlassClassificationPipeline([RandomForestClassifier])]
    with pytest.raises(ValueError, match="All pipelines must have the same problem type."):
        StackedEnsembleClassifier(input_pipelines=input_pipelines)


def test_stacked_ensemble_init_with_multiple_same_estimators(X_y_binary, logistic_regression_binary_pipeline_class):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    input_pipelines = [logistic_regression_binary_pipeline_class(parameters={}),
                       logistic_regression_binary_pipeline_class(parameters={})]
    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        'cv': None,
        'n_jobs': 1
    }
    assert clf.parameters == expected_parameters

    fitted = clf.fit(X, y)
    assert isinstance(fitted, StackedEnsembleClassifier)

    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred.to_series()).all()


def test_stacked_ensemble_n_jobs_negative_one(X_y_binary, logistic_regression_binary_pipeline_class):
    X, y = X_y_binary
    input_pipelines = [logistic_regression_binary_pipeline_class(parameters={})]
    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=-1)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        'cv': None,
        'n_jobs': -1
    }
    assert clf.parameters == expected_parameters
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred.to_series()).all()


@patch('evalml.pipelines.components.ensemble.StackedEnsembleClassifier._stacking_estimator_class')
def test_stacked_ensemble_does_not_overwrite_pipeline_random_seed(mock_stack,
                                                                  logistic_regression_binary_pipeline_class):
    input_pipelines = [logistic_regression_binary_pipeline_class(parameters={}, random_seed=3),
                       logistic_regression_binary_pipeline_class(parameters={}, random_seed=4)]
    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, random_seed=5, n_jobs=1)
    estimators_used_in_ensemble = mock_stack.call_args[1]['estimators']
    assert clf.random_seed == 5
    assert estimators_used_in_ensemble[0][1].pipeline.random_seed == 3
    assert estimators_used_in_ensemble[1][1].pipeline.random_seed == 4


def test_stacked_ensemble_multilevel(logistic_regression_binary_pipeline_class):
    # checks passing a stacked ensemble classifier as a final estimator
    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series([1, 0] * 25)
    base = StackedEnsembleClassifier(input_pipelines=[logistic_regression_binary_pipeline_class(parameters={})], n_jobs=1)
    clf = StackedEnsembleClassifier(input_pipelines=[logistic_regression_binary_pipeline_class(parameters={})],
                                    final_estimator=base,
                                    n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred.to_series()).all()


def test_stacked_problem_types():
    assert ProblemTypes.BINARY in StackedEnsembleClassifier.supported_problem_types
    assert ProblemTypes.MULTICLASS in StackedEnsembleClassifier.supported_problem_types
    assert StackedEnsembleClassifier.supported_problem_types == [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                                                                 ProblemTypes.TIME_SERIES_BINARY,
                                                                 ProblemTypes.TIME_SERIES_MULTICLASS]


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_stacked_fit_predict_classification(X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        num_classes = 2
        pipeline_class = BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        num_classes = 3
        pipeline_class = MulticlassClassificationPipeline
    input_pipelines = [pipeline_class([classifier]) for classifier in stackable_classifiers]
    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, ww.DataColumn)
    assert not np.isnan(y_pred.to_series()).all()

    y_pred_proba = clf.predict_proba(X)
    assert isinstance(y_pred_proba, ww.DataTable)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba.to_dataframe()).all().all()

    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, final_estimator=RandomForestClassifier(), n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, ww.DataColumn)
    assert not np.isnan(y_pred.to_series()).all()

    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert isinstance(y_pred_proba, ww.DataTable)
    assert not np.isnan(y_pred_proba.to_dataframe()).all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@patch('evalml.pipelines.components.ensemble.StackedEnsembleClassifier.fit')
def test_stacked_feature_importance(mock_fit, X_y_binary, X_y_multi, stackable_classifiers, problem_type):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline_class = BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline_class = MulticlassClassificationPipeline
    input_pipelines = [pipeline_class([classifier]) for classifier in stackable_classifiers]
    clf = StackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(NotImplementedError, match="feature_importance is not implemented"):
        clf.feature_importance
