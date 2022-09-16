from unittest.mock import patch

import featuretools as ft
import numpy as np
import pandas as pd
import pytest
from skopt.space import Categorical, Integer, Real

from evalml.automl.automl_algorithm import AutoMLAlgorithmException, IterativeAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    Estimator,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
)
from evalml.pipelines.components import (
    DateTimeFeaturizer,
    EmailFeaturizer,
    NaturalLanguageFeaturizer,
    TimeSeriesFeaturizer,
    URLFeaturizer,
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes, is_time_series


@pytest.fixture
def dummy_binary_pipeline_classes():
    def _method(hyperparameters=["default", "other"]):
        class MockEstimator(Estimator):
            name = "Mock Classifier"
            model_family = ModelFamily.RANDOM_FOREST
            supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
            if isinstance(hyperparameters, (list, tuple, Real, Categorical, Integer)):
                hyperparameter_ranges = {"dummy_parameter": hyperparameters}
            else:
                hyperparameter_ranges = {"dummy_parameter": [hyperparameters]}

            def __init__(
                self, dummy_parameter="default", n_jobs=-1, random_seed=0, **kwargs
            ):
                super().__init__(
                    parameters={
                        "dummy_parameter": dummy_parameter,
                        **kwargs,
                        "n_jobs": n_jobs,
                    },
                    component_obj=None,
                    random_seed=random_seed,
                )

        allowed_component_graphs = {
            "graph_1": [MockEstimator],
            "graph_2": [MockEstimator],
            "graph_3": [MockEstimator],
        }
        return [
            BinaryClassificationPipeline([MockEstimator]),
            BinaryClassificationPipeline([MockEstimator]),
            BinaryClassificationPipeline([MockEstimator]),
        ], allowed_component_graphs

    return _method


def test_iterative_algorithm_init(
    X_y_binary,
):
    X, y = X_y_binary
    algo = IterativeAlgorithm(X=X, y=y, problem_type="binary")
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.default_max_batches == 1
    estimators = get_estimators("binary")
    assert len(algo.allowed_pipelines) == len(
        [
            make_pipeline(
                X,
                y,
                estimator,
                "binary",
            )
            for estimator in estimators
        ],
    )


def test_make_iterative_algorithm_search_parameters_error(
    dummy_binary_pipeline_classes,
    X_y_binary,
):
    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    X, y = X_y_binary

    search_parameters = [
        {"Imputer": {"numeric_imput_strategy": ["median"]}},
        {"One Hot Encoder": {"value1": ["value2"]}},
    ]

    with pytest.raises(
        ValueError,
        match="If search_parameters provided, must be of type dict",
    ):
        IterativeAlgorithm(
            X=X,
            y=y,
            problem_type="binary",
            allowed_component_graphs=allowed_component_graphs,
            search_parameters=search_parameters,
        )


def test_iterative_algorithm_allowed_pipelines(
    X_y_binary,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
    )
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == dummy_binary_pipeline_classes


def test_iterative_algorithm_empty(X_y_binary, dummy_binary_pipeline_classes):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()

    with pytest.raises(ValueError, match="No allowed pipelines to search"):
        IterativeAlgorithm(X=X, y=y, problem_type="binary", allowed_component_graphs={})

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
    )
    algo.allowed_pipelines = []
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []

    next_batch = algo.next_batch()
    assert [p.__class__ for p in next_batch] == []
    assert algo.pipeline_number == 0
    assert algo.batch_number == 1

    with pytest.raises(
        AutoMLAlgorithmException,
        match="No results were reported from the first batch",
    ):
        algo.next_batch()
    assert algo.batch_number == 1
    assert algo.pipeline_number == 0


@pytest.mark.parametrize("ensembling_value", [True, False])
@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_iterative_algorithm_results(
    mock_opt_tell,
    ensembling_value,
    dummy_binary_pipeline_classes,
    X_y_binary,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        ensembling=ensembling_value,
    )
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == dummy_binary_pipeline_classes

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    assert len(next_batch) == len(dummy_binary_pipeline_classes)
    assert [p.__class__ for p in next_batch] == [
        p.__class__ for p in dummy_binary_pipeline_classes
    ]
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert all(
        [p.parameters == p.component_graph.default_parameters for p in next_batch],
    )
    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # subsequent batches contain pipelines_per_batch copies of one pipeline, moving from best to worst from the first batch
    last_batch_number = algo.batch_number
    last_pipeline_number = algo.pipeline_number
    all_parameters = []

    for i in range(1, 5):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert len(next_batch) == algo.pipelines_per_batch
            num_pipelines_classes = (
                (len(dummy_binary_pipeline_classes) + 1)
                if ensembling_value
                else len(dummy_binary_pipeline_classes)
            )
            cls = dummy_binary_pipeline_classes[
                (algo.batch_number - 2) % num_pipelines_classes
            ].__class__
            assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
            assert all(
                [p.parameters["Mock Classifier"]["n_jobs"] == -1 for p in next_batch],
            )
            assert all((p.random_seed == algo.random_seed) for p in next_batch)
            assert algo.pipeline_number == last_pipeline_number + len(next_batch)
            last_pipeline_number = algo.pipeline_number
            assert algo.batch_number == last_batch_number + 1
            last_batch_number = algo.batch_number
            all_parameters.extend([p.parameters for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})

        assert any(
            [p != dummy_binary_pipeline_classes[0].parameters for p in all_parameters],
        )

        if ensembling_value:
            # check next batch is stacking ensemble batch
            assert algo.batch_number == (len(dummy_binary_pipeline_classes) + 1) * i
            next_batch = algo.next_batch()
            assert len(next_batch) == 1
            assert algo.batch_number == last_batch_number + 1
            last_batch_number = algo.batch_number
            assert algo.pipeline_number == last_pipeline_number + 1
            last_pipeline_number = algo.pipeline_number
            scores = np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})
            assert pipeline.model_family == ModelFamily.ENSEMBLE
            assert pipeline.random_seed == algo.random_seed
            estimators_used_in_ensemble = pipeline.component_graph.get_estimators()
            random_seeds_the_same = [
                (estimator.random_seed == algo.random_seed)
                for estimator in estimators_used_in_ensemble
            ]
            assert all(random_seeds_the_same)
            assert ModelFamily.ENSEMBLE not in algo._best_pipeline_info


@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_iterative_algorithm_passes_pipeline_params(
    mock_opt_tell,
    X_y_binary,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        search_parameters={
            "pipeline": {"gap": 2, "max_delay": 10, "forecast_horizon": 3},
        },
    )

    next_batch = algo.next_batch()
    assert all(
        [
            p.parameters["pipeline"]
            == {"gap": 2, "max_delay": 10, "forecast_horizon": 3}
            for p in next_batch
        ],
    )

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 5):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all(
                [
                    p.parameters["pipeline"]
                    == {"gap": 2, "max_delay": 10, "forecast_horizon": 3}
                    for p in next_batch
                ],
            )
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_iterative_algorithm_passes_njobs(
    mock_opt_tell,
    X_y_binary,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        n_jobs=2,
        ensembling=False,
    )
    next_batch = algo.next_batch()

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 3):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all(
                [p.parameters["Mock Classifier"]["n_jobs"] == 2 for p in next_batch],
            )
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@pytest.mark.parametrize("is_regression", [True, False])
@pytest.mark.parametrize("estimator", ["XGBoost", "CatBoost"])
@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_iterative_algorithm_passes_n_jobs_catboost_xgboost(
    mock_opt_tell,
    is_regression,
    estimator,
    X_y_binary,
    X_y_regression,
):
    if is_regression:
        X, y = X_y_regression
        component_graphs = {"graph": [f"{estimator} Regressor"]}
        problem_type = "regression"
    else:
        X, y = X_y_binary
        component_graphs = {"graph": [f"{estimator} Classifier"]}
        problem_type = "binary"

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type=problem_type,
        allowed_component_graphs=component_graphs,
        n_jobs=2,
        ensembling=False,
    )
    next_batch = algo.next_batch()

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for _ in range(1, 3):
        for _ in range(len(component_graphs)):
            next_batch = algo.next_batch()
            for parameter_values in [list(p.parameters.values()) for p in next_batch]:
                assert parameter_values[0]["n_jobs"] == 2
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@pytest.mark.parametrize("ensembling_value", [True, False])
def test_iterative_algorithm_one_allowed_pipeline(
    X_y_binary,
    ensembling_value,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    dummy_binary_pipeline_classes = [dummy_binary_pipeline_classes[0]]
    allowed_component_graphs = {"graph_1": allowed_component_graphs["graph_1"]}
    # Checks that when len(allowed_pipeline) == 1, ensembling is not run, even if set to True
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        ensembling=ensembling_value,
    )
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == dummy_binary_pipeline_classes

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    assert len(next_batch) == 1
    assert [p.__class__ for p in next_batch] == [
        p.__class__ for p in dummy_binary_pipeline_classes
    ]
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert all(
        [p.parameters == p.component_graph.default_parameters for p in next_batch],
    )
    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # subsequent batches contain pipelines_per_batch copies of one pipeline, moving from best to worst from the first batch
    last_batch_number = algo.batch_number
    last_pipeline_number = algo.pipeline_number
    all_parameters = []
    for i in range(1, 5):
        next_batch = algo.next_batch()
        assert len(next_batch) == algo.pipelines_per_batch
        assert all((p.random_seed == algo.random_seed) for p in next_batch)
        assert [p.__class__ for p in next_batch] == [
            dummy_binary_pipeline_classes[0].__class__,
        ] * algo.pipelines_per_batch
        assert algo.pipeline_number == last_pipeline_number + len(next_batch)
        last_pipeline_number = algo.pipeline_number
        assert algo.batch_number == last_batch_number + 1
        last_batch_number = algo.batch_number
        all_parameters.extend([p.parameters for p in next_batch])
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number})

        assert any(
            [
                p
                != dummy_binary_pipeline_classes[0]
                .__class__({})
                .component_graph.default_parameters
                for p in all_parameters
            ],
        )


@pytest.mark.parametrize("text_in_ensembling", [True, False])
@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_binary(
    n_jobs,
    X_y_binary,
    text_in_ensembling,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        ensembling=True,
        text_in_ensembling=text_in_ensembling,
        n_jobs=n_jobs,
    )

    next_batch = algo.next_batch()
    seen_ensemble = False
    scores = range(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(5):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            if isinstance(pipeline.estimator, StackedEnsembleClassifier):
                seen_ensemble = True
                if text_in_ensembling:
                    assert (
                        pipeline.parameters["Stacked Ensemble Classifier"]["n_jobs"]
                        == 1
                    )
                else:
                    assert (
                        pipeline.parameters["Stacked Ensemble Classifier"]["n_jobs"]
                        == n_jobs
                    )
    assert seen_ensemble


@pytest.mark.parametrize("text_in_ensembling", [True, False])
@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_regression(
    n_jobs,
    text_in_ensembling,
    X_y_regression,
    linear_regression_pipeline,
):
    X, y = X_y_regression

    allowed_component_graphs = {
        "graph_1": linear_regression_pipeline.component_graph,
        "graph_2": linear_regression_pipeline.component_graph,
    }
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="regression",
        allowed_component_graphs=allowed_component_graphs,
        ensembling=True,
        text_in_ensembling=text_in_ensembling,
        n_jobs=n_jobs,
    )
    next_batch = algo.next_batch()
    seen_ensemble = False
    scores = range(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(5):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            if isinstance(pipeline.estimator, StackedEnsembleRegressor):
                seen_ensemble = True
                if text_in_ensembling:
                    assert (
                        pipeline.parameters["Stacked Ensemble Regressor"]["n_jobs"] == 1
                    )
                else:
                    assert (
                        pipeline.parameters["Stacked Ensemble Regressor"]["n_jobs"]
                        == n_jobs
                    )
    assert seen_ensemble


@pytest.mark.parametrize(
    "parameters",
    [1, "hello", 1.3, -1.0006],
)
def test_iterative_algorithm_search_params(
    X_y_binary,
    parameters,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes(parameters)

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        random_seed=0,
        search_parameters={
            "pipeline": {"gap": 2, "max_delay": 10, "forecast_horizon": 3},
            "Mock Classifier": {"dummy_parameter": parameters},
        },
    )

    parameter = parameters
    next_batch = algo.next_batch()
    assert all(
        [
            p.parameters["pipeline"]
            == {"gap": 2, "max_delay": 10, "forecast_horizon": 3}
            for p in next_batch
        ],
    )
    assert all(
        [
            p.parameters["Mock Classifier"]
            == {"dummy_parameter": parameter, "n_jobs": -1}
            for p in next_batch
        ],
    )

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # make sure that future batches have the same parameter value
    for i in range(1, 5):
        next_batch = algo.next_batch()
        assert all(
            [
                p.parameters["Mock Classifier"]["dummy_parameter"] == parameter
                for p in next_batch
            ],
        )


def test_iterative_algorithm_search_params_kwargs(
    X_y_binary,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary

    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
        search_parameters={
            "Mock Classifier": {"dummy_parameter": "dummy", "fake_param": "fake"},
        },
        random_seed=0,
    )

    next_batch = algo.next_batch()
    assert all(
        [
            p.parameters["Mock Classifier"]
            == {"dummy_parameter": "dummy", "n_jobs": -1, "fake_param": "fake"}
            for p in next_batch
        ],
    )


def test_iterative_algorithm_results_best_pipeline_info_id(
    X_y_binary,
    dummy_binary_pipeline_classes,
    logistic_regression_component_graph,
):
    X, y = X_y_binary
    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    allowed_component_graphs = {
        "graph_1": allowed_component_graphs["graph_1"],
        "graph_2": logistic_regression_component_graph,
    }
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
    )

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    scores = np.arange(0, len(next_batch))
    for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number + pipeline_num})
    assert algo._best_pipeline_info[ModelFamily.RANDOM_FOREST]["id"] == 3
    assert algo._best_pipeline_info[ModelFamily.LINEAR_MODEL]["id"] == 2

    for i in range(1, 3):
        next_batch = algo.next_batch()
        scores = -np.arange(
            1,
            len(next_batch),
        )  # Score always gets better with each pipeline
        for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
            algo.add_result(
                score,
                pipeline,
                {"id": algo.pipeline_number + pipeline_num},
            )
            assert (
                algo._best_pipeline_info[pipeline.model_family]["id"]
                == algo.pipeline_number + pipeline_num
            )


@pytest.mark.parametrize(
    "problem_type",
    [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS],
)
def test_iterative_algorithm_first_batch_order(problem_type, X_y_binary):
    X, y = X_y_binary

    algo = IterativeAlgorithm(X=X, y=y, problem_type=problem_type)

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    estimators_in_first_batch = [p.estimator.name for p in next_batch]

    if problem_type == ProblemTypes.REGRESSION:
        linear_models = ["Elastic Net Regressor"]
        extra_dep_estimators = [
            "XGBoost Regressor",
            "LightGBM Regressor",
            "CatBoost Regressor",
        ]
        core_estimators = [
            "Random Forest Regressor",
            "Decision Tree Regressor",
            "Extra Trees Regressor",
        ]
    else:
        linear_models = ["Elastic Net Classifier", "Logistic Regression Classifier"]
        extra_dep_estimators = [
            "XGBoost Classifier",
            "LightGBM Classifier",
            "CatBoost Classifier",
        ]
        core_estimators = [
            "Random Forest Classifier",
            "Decision Tree Classifier",
            "Extra Trees Classifier",
        ]
    assert (
        estimators_in_first_batch
        == linear_models + extra_dep_estimators + core_estimators
    )


def test_iterative_algorithm_first_batch_order_param(X_y_binary):
    X, y = X_y_binary

    # put random forest first
    estimator_family_order = [
        ModelFamily.RANDOM_FOREST,
        ModelFamily.LINEAR_MODEL,
        ModelFamily.DECISION_TREE,
        ModelFamily.EXTRA_TREES,
        ModelFamily.XGBOOST,
        ModelFamily.LIGHTGBM,
        ModelFamily.CATBOOST,
    ]
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        _estimator_family_order=estimator_family_order,
    )
    next_batch = algo.next_batch()
    estimators_in_first_batch = [p.estimator.name for p in next_batch]

    final_estimators = [
        "XGBoost Classifier",
        "LightGBM Classifier",
        "CatBoost Classifier",
    ]
    assert (
        estimators_in_first_batch
        == [
            "Random Forest Classifier",
            "Elastic Net Classifier",
            "Logistic Regression Classifier",
            "Decision Tree Classifier",
            "Extra Trees Classifier",
        ]
        + final_estimators
    )


@pytest.mark.parametrize(
    "sampler",
    ["Undersampler", "Oversampler"],
)
@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_iterative_algorithm_sampling_params(
    problem_type,
    sampler,
    mock_imbalanced_data_X_y,
):
    X, y = mock_imbalanced_data_X_y(problem_type, "some", "small")
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type=problem_type,
        random_seed=0,
        sampler_name=sampler,
    )
    next_batch = algo.next_batch()
    for p in next_batch:
        for component in p.component_graph:
            if "sampler" in component.name:
                assert component.parameters["sampling_ratio"] == 0.25

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        for p in next_batch:
            for component in p.component_graph:
                if "sampler" in component.name:
                    assert component.parameters["sampling_ratio"] == 0.25


@pytest.mark.parametrize("allowed_model_families", [None, [ModelFamily.XGBOOST]])
@pytest.mark.parametrize(
    "allowed_component_graphs",
    [None, {"Pipeline_1": ["Imputer", "XGBoost Classifier"]}],
)
@pytest.mark.parametrize("allow_long_running_models", [True, False])
@pytest.mark.parametrize(
    "length,models_missing",
    [
        (10, []),
        (75, []),
        (100, ["Elastic Net Classifier", "XGBoost Classifier"]),
        (160, ["Elastic Net Classifier", "XGBoost Classifier", "CatBoost Classifier"]),
    ],
)
def test_iterative_algorithm_allow_long_running_models(
    length,
    models_missing,
    allow_long_running_models,
    allowed_component_graphs,
    allowed_model_families,
):
    X = pd.DataFrame()
    y = pd.Series([i for i in range(length)] * 5)
    y_short = pd.Series([i for i in range(10)] * 5)
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="multiclass",
        random_seed=0,
        allowed_model_families=allowed_model_families,
        allowed_component_graphs=allowed_component_graphs,
        allow_long_running_models=allow_long_running_models,
    )
    if allowed_model_families is not None or allowed_component_graphs is not None:
        assert len(algo.allowed_pipelines) == 1
        return
    algo_short = IterativeAlgorithm(
        X=X,
        y=y_short,
        problem_type="multiclass",
        random_seed=0,
        allowed_model_families=allowed_model_families,
        allowed_component_graphs=allowed_component_graphs,
    )
    if allow_long_running_models:
        assert len(algo_short.allowed_pipelines) == len(algo.allowed_pipelines)
    else:
        assert len(algo_short.allowed_pipelines) == len(algo.allowed_pipelines) + len(
            models_missing,
        )
        for p in algo.allowed_pipelines:
            assert all([s not in p.name for s in models_missing])


@pytest.mark.parametrize("problem", ["binary", "multiclass", "regression"])
@pytest.mark.parametrize("allow_long_running_models", [True, False])
@pytest.mark.parametrize(
    "length,models_missing",
    [(10, 0), (75, 0), (100, 2), (160, 3)],
)
def test_iterative_algorithm_allow_long_running_models_problem(
    length,
    models_missing,
    allow_long_running_models,
    problem,
):
    X = pd.DataFrame()
    y = pd.Series([i for i in range(length)] * 5)
    y_short = pd.Series([i for i in range(10)] * 5)
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type=problem,
        random_seed=0,
        allow_long_running_models=allow_long_running_models,
    )
    algo_reg = IterativeAlgorithm(
        X=X,
        y=y_short,
        problem_type=problem,
        random_seed=0,
    )
    if problem != "multiclass" or allow_long_running_models:
        assert len(algo.allowed_pipelines) == len(algo_reg.allowed_pipelines)
        models_missing = 0

    assert len(algo.allowed_pipelines) + models_missing == len(
        algo_reg.allowed_pipelines,
    )


def test_iterative_algorithm_allow_long_running_models_next_batch():
    models_missing = [
        "Elastic Net Classifier",
        "XGBoost Classifier",
        "CatBoost Classifier",
    ]
    X = pd.DataFrame()
    y = pd.Series([i for i in range(200)] * 5)

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="multiclass",
        random_seed=0,
        allow_long_running_models=False,
    )
    next_batch = algo.next_batch()

    for pipeline in next_batch:
        assert all([m not in pipeline.name for m in models_missing])

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 5):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            assert all([m not in pipeline.name for m in models_missing])
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@patch("evalml.tuners.skopt_tuner.Optimizer.tell")
def test_iterative_algorithm_passes_features(
    mock_opt_tell,
    X_y_binary,
    dummy_binary_pipeline_classes,
):
    X, y = X_y_binary
    X = pd.DataFrame(X)  # Drop ww information since setting column types fails
    X.columns = X.columns.astype(str)
    X_transform = X.iloc[len(X) // 3 :]

    es = ft.EntitySet()
    es = es.add_dataframe(
        dataframe_name="X",
        dataframe=X_transform,
        index="index",
        make_index=True,
    )
    _, features = ft.dfs(
        entityset=es,
        target_dataframe_name="X",
        trans_primitives=["absolute"],
    )

    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        search_parameters={"DFS Transformer": {"features": features}},
        ensembling=False,
        features=features,
    )
    next_batch = algo.next_batch()
    assert all(
        ["DFS Transformer" in p.component_graph.compute_order for p in next_batch],
    )
    assert all(
        [p.parameters["DFS Transformer"]["features"] == features for p in next_batch],
    )

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 3):
        next_batch = algo.next_batch()
        assert all(
            ["DFS Transformer" in p.component_graph.compute_order for p in next_batch],
        )
        assert all(
            [
                p.parameters["DFS Transformer"]["features"] == features
                for p in next_batch
            ],
        )
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number})


def test_iterative_algorithm_add_result_cache(
    X_y_binary,
    dummy_binary_pipeline_classes,
    logistic_regression_component_graph,
):
    X, y = X_y_binary
    (
        dummy_binary_pipeline_classes,
        allowed_component_graphs,
    ) = dummy_binary_pipeline_classes()
    allowed_component_graphs = {
        "graph_1": allowed_component_graphs["graph_1"],
        "graph_2": logistic_regression_component_graph,
    }
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
        allowed_component_graphs=allowed_component_graphs,
    )

    cache = {"some_cache_key": "some_cache_value"}
    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    scores = np.arange(0, len(next_batch))
    for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
        algo.add_result(
            score,
            pipeline,
            {"id": algo.pipeline_number + pipeline_num},
            cached_data=cache,
        )

    for values in algo._best_pipeline_info.values():
        assert values["cached_data"] == cache


def test_iterative_algorithm_num_pipelines_per_batch(X_y_binary):
    X, y = X_y_binary
    algo = IterativeAlgorithm(
        X=X,
        y=y,
        problem_type="binary",
    )

    for i in range(4):
        if i == 0:
            assert algo.num_pipelines_per_batch(i) == len(algo.allowed_pipelines)
        else:
            assert algo.num_pipelines_per_batch(i) == algo.pipelines_per_batch


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_exclude_featurizers_iterative_algorithm(
    problem_type,
    input_type,
    get_test_data_from_configuration,
):
    parameters = {}
    if is_time_series(problem_type):
        parameters = {
            "time_index": "dates",
            "gap": 1,
            "max_delay": 1,
            "forecast_horizon": 3,
        }

    X, y = get_test_data_from_configuration(
        input_type,
        problem_type,
        column_names=["dates", "text", "email", "url"],
    )

    algo = IterativeAlgorithm(
        X,
        y,
        problem_type,
        sampler_name=None,
        search_parameters={"pipeline": parameters},
        exclude_featurizers=[
            "DatetimeFeaturizer",
            "EmailFeaturizer",
            "URLFeaturizer",
            "NaturalLanguageFeaturizer",
            "TimeSeriesFeaturizer",
        ],
    )

    pipelines = [pl for pl in algo.allowed_pipelines]

    # A check to make sure we actually retrieve constructed pipelines from the algo.
    assert len(pipelines) > 0

    assert not any(
        [
            DateTimeFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [EmailFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [URLFeaturizer.name in pl.component_graph.compute_order for pl in pipelines],
    )
    assert not any(
        [
            NaturalLanguageFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
    assert not any(
        [
            TimeSeriesFeaturizer.name in pl.component_graph.compute_order
            for pl in pipelines
        ],
    )
