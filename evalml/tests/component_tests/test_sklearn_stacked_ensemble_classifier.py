import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
)
from evalml.pipelines.components import (
    BaselineClassifier,
    RandomForestClassifier,
)
from evalml.pipelines.components.ensemble import (
    SklearnStackedEnsembleClassifier,
)
from evalml.problem_types import ProblemTypes


def test_sklearn_stacked_model_family():
    assert SklearnStackedEnsembleClassifier.model_family == ModelFamily.ENSEMBLE


def test_sklearn_stacked_default_parameters():
    assert SklearnStackedEnsembleClassifier.default_parameters == {
        "final_estimator": None,
        "cv": None,
        "n_jobs": -1,
    }


def test_sklearn_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(
        EnsembleMissingPipelinesError, match="must not be None or an empty list."
    ):
        SklearnStackedEnsembleClassifier()
    with pytest.raises(
        EnsembleMissingPipelinesError, match="must not be None or an empty list."
    ):
        SklearnStackedEnsembleClassifier(input_pipelines=[])


def test_sklearn_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(
        ValueError,
        match="Pipelines with any of the following model families cannot be used as base pipelines",
    ):
        SklearnStackedEnsembleClassifier(
            input_pipelines=[BinaryClassificationPipeline([BaselineClassifier])]
        )


def test_sklearn_stacked_different_input_pipelines_classification():
    input_pipelines = [
        BinaryClassificationPipeline([RandomForestClassifier]),
        MulticlassClassificationPipeline([RandomForestClassifier]),
    ]
    with pytest.raises(
        ValueError, match="All pipelines must have the same problem type."
    ):
        SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines)


def test_sklearn_stacked_ensemble_init_with_multiple_same_estimators(
    X_y_binary, logistic_regression_binary_pipeline_class
):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_binary
    input_pipelines = [
        logistic_regression_binary_pipeline_class(parameters={}),
        logistic_regression_binary_pipeline_class(parameters={}),
    ]
    clf = SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        "cv": None,
        "n_jobs": 1,
    }
    assert clf.parameters == expected_parameters

    fitted = clf.fit(X, y)
    assert isinstance(fitted, SklearnStackedEnsembleClassifier)

    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_sklearn_stacked_ensemble_n_jobs_negative_one(
    X_y_binary, logistic_regression_binary_pipeline_class
):
    X, y = X_y_binary
    input_pipelines = [logistic_regression_binary_pipeline_class(parameters={})]
    clf = SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=-1)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        "cv": None,
        "n_jobs": -1,
    }
    assert clf.parameters == expected_parameters
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


@patch(
    "evalml.pipelines.components.ensemble.SklearnStackedEnsembleClassifier._stacking_estimator_class"
)
def test_sklearn_stacked_ensemble_does_not_overwrite_pipeline_random_seed(
    mock_stack, logistic_regression_binary_pipeline_class
):
    input_pipelines = [
        logistic_regression_binary_pipeline_class(parameters={}, random_seed=3),
        logistic_regression_binary_pipeline_class(parameters={}, random_seed=4),
    ]
    clf = SklearnStackedEnsembleClassifier(
        input_pipelines=input_pipelines, random_seed=5, n_jobs=1
    )
    estimators_used_in_ensemble = mock_stack.call_args[1]["estimators"]
    assert clf.random_seed == 5
    assert estimators_used_in_ensemble[0][1].pipeline.random_seed == 3
    assert estimators_used_in_ensemble[1][1].pipeline.random_seed == 4


def test_sklearn_stacked_ensemble_multilevel(logistic_regression_binary_pipeline_class):
    # checks passing a stacked ensemble classifier as a final estimator
    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series([1, 0] * 25)
    base = SklearnStackedEnsembleClassifier(
        input_pipelines=[logistic_regression_binary_pipeline_class(parameters={})],
        n_jobs=1,
    )
    clf = SklearnStackedEnsembleClassifier(
        input_pipelines=[logistic_regression_binary_pipeline_class(parameters={})],
        final_estimator=base,
        n_jobs=1,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_sklearn_stacked_problem_types():
    assert (
        ProblemTypes.BINARY in SklearnStackedEnsembleClassifier.supported_problem_types
    )
    assert (
        ProblemTypes.MULTICLASS
        in SklearnStackedEnsembleClassifier.supported_problem_types
    )
    assert SklearnStackedEnsembleClassifier.supported_problem_types == [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_sklearn_stacked_fit_predict_classification(
    X_y_binary, X_y_multi, stackable_classifiers, problem_type
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        num_classes = 2
        pipeline_class = BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        num_classes = 3
        pipeline_class = MulticlassClassificationPipeline
    input_pipelines = [
        pipeline_class([classifier]) for classifier in stackable_classifiers
    ]
    clf = SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    y_pred_proba = clf.predict_proba(X)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert not np.isnan(y_pred_proba).all().all()

    clf = SklearnStackedEnsembleClassifier(
        input_pipelines=input_pipelines,
        final_estimator=RandomForestClassifier(),
        n_jobs=1,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    y_pred_proba = clf.predict_proba(X)
    assert y_pred_proba.shape == (len(y), num_classes)
    assert isinstance(y_pred_proba, pd.DataFrame)
    assert not np.isnan(y_pred_proba).all().all()


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
@patch("evalml.pipelines.components.ensemble.SklearnStackedEnsembleClassifier.fit")
def test_sklearn_stacked_feature_importance(
    mock_fit, X_y_binary, X_y_multi, stackable_classifiers, problem_type
):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_binary
        pipeline_class = BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_multi
        pipeline_class = MulticlassClassificationPipeline
    input_pipelines = [
        pipeline_class([classifier]) for classifier in stackable_classifiers
    ]
    clf = SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(
        NotImplementedError, match="feature_importance is not implemented"
    ):
        clf.feature_importance


def test_sklearn_stacked_deprecation_warning(logistic_regression_binary_pipeline_class):
    input_pipelines = [logistic_regression_binary_pipeline_class(parameters={})]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SklearnStackedEnsembleClassifier(input_pipelines=input_pipelines)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert (
            str(w[0].message)
            == "Scikit-learn based ensemblers will be completely removed in the next release. Utilize the new `StackedEnsembleRegressor` or `StackedEnsembleClassifier` ensembler instead."
        )
