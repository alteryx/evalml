import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from evalml.exceptions import EnsembleMissingPipelinesError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline, RegressionPipeline
from evalml.pipelines.components import (
    BaselineRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    SklearnStackedEnsembleRegressor,
)
from evalml.problem_types import ProblemTypes


def test_sklearn_stacked_model_family():
    assert SklearnStackedEnsembleRegressor.model_family == ModelFamily.ENSEMBLE


def test_sklearn_stacked_default_parameters():
    assert SklearnStackedEnsembleRegressor.default_parameters == {
        "final_estimator": None,
        "cv": None,
        "n_jobs": -1,
    }


def test_sklearn_stacked_ensemble_init_with_invalid_estimators_parameter():
    with pytest.raises(
        EnsembleMissingPipelinesError, match="must not be None or an empty list."
    ):
        SklearnStackedEnsembleRegressor()
    with pytest.raises(
        EnsembleMissingPipelinesError, match="must not be None or an empty list."
    ):
        SklearnStackedEnsembleRegressor(input_pipelines=[])


def test_sklearn_stacked_ensemble_nonstackable_model_families():
    with pytest.raises(
        ValueError,
        match="Pipelines with any of the following model families cannot be used as base pipelines",
    ):
        SklearnStackedEnsembleRegressor(
            input_pipelines=[RegressionPipeline([BaselineRegressor])]
        )


def test_sklearn_stacked_different_input_pipelines_regression():
    input_pipelines = [
        RegressionPipeline([RandomForestRegressor]),
        BinaryClassificationPipeline([RandomForestClassifier]),
    ]
    with pytest.raises(
        ValueError, match="All pipelines must have the same problem type."
    ):
        SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines)


def test_sklearn_stacked_ensemble_init_with_multiple_same_estimators(
    X_y_regression, linear_regression_pipeline_class
):
    # Checks that it is okay to pass multiple of the same type of estimator
    X, y = X_y_regression
    input_pipelines = [
        linear_regression_pipeline_class(parameters={}),
        linear_regression_pipeline_class(parameters={}),
    ]
    clf = SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines, n_jobs=1)
    expected_parameters = {
        "input_pipelines": input_pipelines,
        "final_estimator": None,
        "cv": None,
        "n_jobs": 1,
    }
    assert clf.parameters == expected_parameters

    fitted = clf.fit(X, y)
    assert isinstance(fitted, SklearnStackedEnsembleRegressor)

    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_sklearn_stacked_ensemble_n_jobs_negative_one(
    X_y_regression, linear_regression_pipeline_class
):
    X, y = X_y_regression
    input_pipelines = [linear_regression_pipeline_class(parameters={})]
    clf = SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines)
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
    "evalml.pipelines.components.ensemble.SklearnStackedEnsembleRegressor._stacking_estimator_class"
)
def test_sklearn_stacked_ensemble_does_not_overwrite_pipeline_random_seed(
    mock_stack, linear_regression_pipeline_class
):
    input_pipelines = [
        linear_regression_pipeline_class(parameters={}, random_seed=3),
        linear_regression_pipeline_class(parameters={}, random_seed=4),
    ]
    clf = SklearnStackedEnsembleRegressor(
        input_pipelines=input_pipelines, random_seed=5, n_jobs=1
    )
    estimators_used_in_ensemble = mock_stack.call_args[1]["estimators"]
    assert clf.random_seed == 5
    assert estimators_used_in_ensemble[0][1].pipeline.random_seed == 3
    assert estimators_used_in_ensemble[1][1].pipeline.random_seed == 4


def test_sklearn_stacked_ensemble_multilevel(linear_regression_pipeline_class):
    # checks passing a stacked ensemble classifier as a final estimator
    X = pd.DataFrame(np.random.rand(50, 5))
    y = pd.Series(
        np.random.rand(
            50,
        )
    )
    base = SklearnStackedEnsembleRegressor(
        input_pipelines=[linear_regression_pipeline_class(parameters={})], n_jobs=1
    )
    clf = SklearnStackedEnsembleRegressor(
        input_pipelines=[linear_regression_pipeline_class(parameters={})],
        final_estimator=base,
        n_jobs=1,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert not np.isnan(y_pred).all()


def test_sklearn_stacked_problem_types():
    assert (
        ProblemTypes.REGRESSION
        in SklearnStackedEnsembleRegressor.supported_problem_types
    )
    assert len(SklearnStackedEnsembleRegressor.supported_problem_types) == 2


def test_sklearn_stacked_fit_predict_regression(X_y_regression, stackable_regressors):
    X, y = X_y_regression
    input_pipelines = [
        RegressionPipeline([regressor]) for regressor in stackable_regressors
    ]
    clf = SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()

    clf = SklearnStackedEnsembleRegressor(
        input_pipelines=input_pipelines,
        final_estimator=RandomForestRegressor(),
        n_jobs=1,
    )
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert len(y_pred) == len(y)
    assert isinstance(y_pred, pd.Series)
    assert not np.isnan(y_pred).all()


@patch("evalml.pipelines.components.ensemble.SklearnStackedEnsembleRegressor.fit")
def test_sklearn_stacked_feature_importance(
    mock_fit, X_y_regression, stackable_regressors
):
    X, y = X_y_regression
    input_pipelines = [
        RegressionPipeline([regressor]) for regressor in stackable_regressors
    ]
    clf = SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines, n_jobs=1)
    clf.fit(X, y)
    mock_fit.assert_called()
    clf._is_fitted = True
    with pytest.raises(
        NotImplementedError, match="feature_importance is not implemented"
    ):
        clf.feature_importance


def test_sklearn_stacked_deprecation_warning(
    linear_regression_pipeline_class,
):
    input_pipelines = [linear_regression_pipeline_class(parameters={})]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SklearnStackedEnsembleRegressor(input_pipelines=input_pipelines)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert (
            str(w[0].message)
            == "Scikit-learn based ensemblers will be completely removed in the next release. Utilize the new `StackedEnsembleRegressor` or `StackedEnsembleClassifier` ensembler instead."
        )
