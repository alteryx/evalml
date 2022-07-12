from unittest.mock import MagicMock

import pytest

from evalml import AutoMLSearch
from evalml.model_family import ModelFamily
from evalml.pipelines import RegressionPipeline
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes


def test_automl_component_graphs_no_allowed_component_graphs_iterative(X_y_regression):
    X, y = X_y_regression
    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="regression",
            allowed_component_graphs=None,
            allowed_model_families=[],
            automl_algorithm="iterative",
        )


def test_automl_allowed_component_graphs_specified_component_graphs_iterative(
    AutoMLTestEnv,
    dummy_regressor_estimator_class,
    dummy_regression_pipeline,
    X_y_regression,
):
    X, y = X_y_regression

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_component_graphs={
            "Mock Regression Pipeline": [dummy_regressor_estimator_class],
        },
        allowed_model_families=None,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("regression")
    expected_component_graph = dummy_regression_pipeline.component_graph
    expected_name = dummy_regression_pipeline.name
    expected_oarameters = dummy_regression_pipeline.parameters
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_oarameters
    assert automl.allowed_model_families == [ModelFamily.NONE]

    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_oarameters
    assert automl.allowed_model_families == [ModelFamily.NONE]


def test_automl_allowed_component_graphs_specified_allowed_model_families_iterative(
    AutoMLTestEnv,
    X_y_regression,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_component_graphs=None,
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.REGRESSION)
        for estimator in get_estimators(
            ProblemTypes.REGRESSION,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_component_graphs=None,
        allowed_model_families=["random_forest"],
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.REGRESSION)
        for estimator in get_estimators(
            ProblemTypes.REGRESSION,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_allowed_component_graphs_init_allowed_both_not_specified_iterative(
    AutoMLTestEnv,
    X_y_regression,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_component_graphs=None,
        allowed_model_families=None,
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.REGRESSION)
        for estimator in get_estimators(ProblemTypes.REGRESSION, model_families=None)
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in expected_pipelines],
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_allowed_component_graphs_init_allowed_both_specified_iterative(
    AutoMLTestEnv,
    dummy_regressor_estimator_class,
    dummy_regression_pipeline,
    X_y_regression,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_regression
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        allowed_component_graphs={
            "Mock Regression Pipeline": [dummy_regressor_estimator_class],
        },
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        automl_algorithm="iterative",
    )
    expected_pipelines = [dummy_regression_pipeline]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in expected_pipelines],
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_allowed_component_graphs_search_iterative(
    AutoMLTestEnv,
    example_regression_graph,
    X_y_regression,
):
    X, y = X_y_regression
    component_graph = {"CG": example_regression_graph}

    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="regression",
        max_iterations=2,
        start_iteration_callback=start_iteration_callback,
        allowed_component_graphs=component_graph,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv("regression")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()

    assert start_iteration_callback.call_count == 2
    assert isinstance(
        start_iteration_callback.call_args_list[0][0][0],
        RegressionPipeline,
    )
    assert isinstance(
        start_iteration_callback.call_args_list[1][0][0],
        RegressionPipeline,
    )
