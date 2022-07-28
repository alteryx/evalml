from unittest.mock import MagicMock

import pandas as pd
import pytest

from evalml import AutoMLSearch
from evalml.automl.callbacks import raise_error_callback
from evalml.automl.utils import get_best_sampler_for_data
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MulticlassClassificationPipeline,
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes


@pytest.mark.parametrize("automl_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_automl_allowed_component_graphs_no_component_graphs(
    automl_type,
    X_y_binary,
    X_y_multi,
):
    is_multiclass = automl_type == ProblemTypes.MULTICLASS
    X, y = X_y_multi if is_multiclass else X_y_binary
    problem_type = "multiclass" if is_multiclass else "binary"
    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type=problem_type,
            allowed_component_graphs=None,
            allowed_model_families=[],
            automl_algorithm="iterative",
        )


def test_automl_component_graphs_specified_component_graphs_binary(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        optimize_thresholds=False,
        allowed_model_families=None,
        automl_algorithm="iterative",
    )
    expected_component_graph = dummy_binary_pipeline.component_graph
    expected_name = dummy_binary_pipeline.name
    expected_parameters = dummy_binary_pipeline.parameters
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]


def test_automl_component_graphs_specified_component_graphs_multi(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_multiclass_pipeline,
    X_y_multi,
):
    X, y = X_y_multi
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs={
            "Mock Multiclass Classification Pipeline": [
                dummy_classifier_estimator_class,
            ],
        },
        allowed_model_families=None,
        automl_algorithm="iterative",
    )
    expected_pipeline = dummy_multiclass_pipeline
    expected_component_graph = expected_pipeline.component_graph
    expected_name = expected_pipeline.name
    expected_parameters = expected_pipeline.parameters
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]

    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    env.mock_fit.assert_called()
    env.mock_score.assert_called()
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]


def test_automl_component_graphs_specified_allowed_model_families_binary(
    AutoMLTestEnv,
    X_y_binary,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs=None,
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.BINARY)
        for estimator in get_estimators(
            ProblemTypes.BINARY,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    env.mock_fit.assert_called()
    env.mock_score.assert_called()

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs=None,
        allowed_model_families=["random_forest"],
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.BINARY)
        for estimator in get_estimators(
            ProblemTypes.BINARY,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    with env.test_context(score_return_value={automl.objective.name: 1.0}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_component_graphs_specified_allowed_model_families_multi(
    AutoMLTestEnv,
    X_y_multi,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_multi
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=None,
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS)
        for estimator in get_estimators(
            ProblemTypes.MULTICLASS,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)

    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    env.mock_fit.assert_called()
    env.mock_score.assert_called()

    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=None,
        allowed_model_families=["random_forest"],
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS)
        for estimator in get_estimators(
            ProblemTypes.MULTICLASS,
            model_families=[ModelFamily.RANDOM_FOREST],
        )
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set([ModelFamily.RANDOM_FOREST])
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_component_graphs_init_allowed_both_not_specified_binary(
    AutoMLTestEnv,
    X_y_binary,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs=None,
        allowed_model_families=None,
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.BINARY)
        for estimator in get_estimators(ProblemTypes.BINARY, model_families=None)
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in expected_pipelines],
    )
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_component_graphs_init_allowed_both_not_specified_multi(
    AutoMLTestEnv,
    X_y_multi,
    assert_allowed_pipelines_equal_helper,
):
    X, y = X_y_multi
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs=None,
        allowed_model_families=None,
        automl_algorithm="iterative",
    )
    expected_pipelines = [
        make_pipeline(X, y, estimator, ProblemTypes.MULTICLASS)
        for estimator in get_estimators(ProblemTypes.MULTICLASS, model_families=None)
    ]
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert_allowed_pipelines_equal_helper(automl.allowed_pipelines, expected_pipelines)
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in expected_pipelines],
    )
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_component_graphs_init_allowed_both_specified_binary(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_binary_pipeline,
    X_y_binary,
):
    X, y = X_y_binary
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="binary",
        allowed_component_graphs={
            "Mock Binary Classification Pipeline": [dummy_classifier_estimator_class],
        },
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    expected_component_graph = dummy_binary_pipeline.component_graph
    expected_name = dummy_binary_pipeline.name
    expected_parameters = dummy_binary_pipeline.parameters
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]

    env = AutoMLTestEnv("binary")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in [dummy_binary_pipeline]],
    )
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


def test_automl_component_graphs_init_allowed_both_specified_multi(
    AutoMLTestEnv,
    dummy_classifier_estimator_class,
    dummy_multiclass_pipeline,
    X_y_multi,
):
    X, y = X_y_multi
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        problem_type="multiclass",
        allowed_component_graphs={
            "Mock Multiclass Classification Pipeline": [
                dummy_classifier_estimator_class,
            ],
        },
        allowed_model_families=[ModelFamily.RANDOM_FOREST],
        automl_algorithm="iterative",
    )
    expected_component_graph = dummy_multiclass_pipeline.component_graph
    expected_name = dummy_multiclass_pipeline.name
    expected_parameters = dummy_multiclass_pipeline.parameters
    assert automl.allowed_pipelines[0].component_graph == expected_component_graph
    assert automl.allowed_pipelines[0].name == expected_name
    assert automl.allowed_pipelines[0].parameters == expected_parameters
    assert automl.allowed_model_families == [ModelFamily.NONE]

    env = AutoMLTestEnv("multiclass")
    with env.test_context(score_return_value={automl.objective.name: 1}):
        automl.search()
    assert set(automl.allowed_model_families) == set(
        [p.model_family for p in [dummy_multiclass_pipeline]],
    )
    env.mock_fit.assert_called()
    env.mock_score.assert_called()


@pytest.mark.parametrize("problem_type", ["binary", "multiclass"])
def test_automl_allowed_component_graphs_search(
    problem_type,
    example_graph,
    X_y_binary,
    X_y_multi,
    AutoMLTestEnv,
):
    if problem_type == "binary":
        X, y = X_y_binary
        score_return_value = {"Log Loss Binary": 1.0}
        expected_mock_class = BinaryClassificationPipeline
    else:
        X, y = X_y_multi
        score_return_value = {"Log Loss Multiclass": 1.0}
        expected_mock_class = MulticlassClassificationPipeline
    component_graph = {"CG": example_graph}

    start_iteration_callback = MagicMock()
    automl = AutoMLSearch(
        X_train=X,
        y_train=y,
        max_iterations=5,
        problem_type=problem_type,
        start_iteration_callback=start_iteration_callback,
        allowed_component_graphs=component_graph,
        optimize_thresholds=False,
        automl_algorithm="iterative",
    )
    env = AutoMLTestEnv(problem_type)
    with env.test_context(score_return_value=score_return_value):
        automl.search()

    assert isinstance(
        start_iteration_callback.call_args_list[0][0][0],
        expected_mock_class,
    )
    for i in range(1, 5):
        if problem_type == "binary":
            assert isinstance(
                start_iteration_callback.call_args_list[i][0][0],
                BinaryClassificationPipeline,
            )
        elif problem_type == "multiclass":
            assert isinstance(
                start_iteration_callback.call_args_list[i][0][0],
                MulticlassClassificationPipeline,
            )


def test_automl_oversampler_selection():
    X = pd.DataFrame({"a": ["a"] * 50 + ["b"] * 25 + ["c"] * 25, "b": list(range(100))})
    y = pd.Series([1] * 90 + [0] * 10)
    X.ww.init(logical_types={"a": "Categorical"})

    sampler = get_best_sampler_for_data(
        X,
        y,
        sampler_method="Oversampler",
        sampler_balanced_ratio=0.5,
    )

    allowed_component_graph = {
        "DropCols": ["Drop Columns Transformer", "X", "y"],
        "Oversampler": [sampler, "DropCols.x", "y"],
        "RF": ["Random Forest Classifier", "Oversampler.x", "Oversampler.y"],
    }

    automl = AutoMLSearch(
        X,
        y,
        problem_type="binary",
        allowed_component_graphs={"pipeline": allowed_component_graph},
        search_parameters={"DropCols": {"columns": ["a"]}},
        error_callback=raise_error_callback,
        automl_algorithm="iterative",
    )
    # This should run without error
    automl.search()
