from unittest.mock import patch

import numpy as np
import pytest
from skopt.space import Categorical, Integer, Real

from evalml.automl.automl_algorithm import (
    AutoMLAlgorithmException,
    IterativeAlgorithm
)
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor
)
from evalml.pipelines.components import Estimator
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes


def test_iterative_algorithm_init_iterative():
    IterativeAlgorithm()


def test_iterative_algorithm_init():
    algo = IterativeAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []


def test_iterative_algorithm_allowed_pipelines(logistic_regression_binary_pipeline_class):
    allowed_pipelines = [logistic_regression_binary_pipeline_class({})]
    algo = IterativeAlgorithm(allowed_pipelines=allowed_pipelines)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == allowed_pipelines


@pytest.fixture
def dummy_binary_pipeline_classes():
    def _method(hyperparameters=['default', 'other']):
        class MockEstimator(Estimator):
            name = "Mock Classifier"
            model_family = ModelFamily.RANDOM_FOREST
            supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
            if isinstance(hyperparameters, (list, tuple, Real, Categorical, Integer)):
                hyperparameter_ranges = {'dummy_parameter': hyperparameters}
            else:
                hyperparameter_ranges = {'dummy_parameter': [hyperparameters]}

            def __init__(self, dummy_parameter='default', n_jobs=-1, random_seed=0, **kwargs):
                super().__init__(parameters={'dummy_parameter': dummy_parameter, **kwargs,
                                             'n_jobs': n_jobs},
                                 component_obj=None, random_seed=random_seed)

        return [BinaryClassificationPipeline([MockEstimator]),
                BinaryClassificationPipeline([MockEstimator]),
                BinaryClassificationPipeline([MockEstimator])]
    return _method


def test_iterative_algorithm_empty(dummy_binary_pipeline_classes):
    algo = IterativeAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []

    next_batch = algo.next_batch()
    assert [p.__class__ for p in next_batch] == []
    assert algo.pipeline_number == 0
    assert algo.batch_number == 1

    with pytest.raises(AutoMLAlgorithmException, match='No results were reported from the first batch'):
        algo.next_batch()
    assert algo.batch_number == 1
    assert algo.pipeline_number == 0


@pytest.mark.parametrize("ensembling_value", [True, False])
@patch('evalml.pipelines.components.ensemble.StackedEnsembleClassifier._stacking_estimator_class')
def test_iterative_algorithm_results(mock_stack, ensembling_value, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, ensembling=ensembling_value)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == dummy_binary_pipeline_classes

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    assert len(next_batch) == len(dummy_binary_pipeline_classes)
    assert [p.__class__ for p in next_batch] == [p.__class__ for p in dummy_binary_pipeline_classes]
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert all([p.parameters == p.default_parameters for p in next_batch])
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
            num_pipelines_classes = (len(dummy_binary_pipeline_classes) + 1) if ensembling_value else len(dummy_binary_pipeline_classes)
            cls = dummy_binary_pipeline_classes[(algo.batch_number - 2) % num_pipelines_classes].__class__
            assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
            assert all([p.parameters['Mock Classifier']['n_jobs'] == -1 for p in next_batch])
            assert all((p.random_seed == algo.random_seed) for p in next_batch)
            assert algo.pipeline_number == last_pipeline_number + len(next_batch)
            last_pipeline_number = algo.pipeline_number
            assert algo.batch_number == last_batch_number + 1
            last_batch_number = algo.batch_number
            all_parameters.extend([p.parameters for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})

        assert any([p != dummy_binary_pipeline_classes[0].parameters for p in all_parameters])

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
            stack_args = mock_stack.call_args[1]['estimators']
            estimators_used_in_ensemble = [args[1] for args in stack_args]
            random_seeds_the_same = [(estimator.pipeline.random_seed == algo.random_seed)
                                     for estimator in estimators_used_in_ensemble]
            assert all(random_seeds_the_same)
            assert ModelFamily.ENSEMBLE not in algo._best_pipeline_info


@pytest.mark.parametrize("ensembling_value", [True, False])
@patch('evalml.pipelines.components.ensemble.StackedEnsembleClassifier._stacking_estimator_class')
def test_iterative_algorithm_passes_pipeline_params(mock_stack, ensembling_value, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, ensembling=ensembling_value,
                              pipeline_params={'pipeline': {"gap": 2, "max_delay": 10}})

    next_batch = algo.next_batch()
    assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 5):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})

        if ensembling_value:
            next_batch = algo.next_batch()
            input_pipelines = next_batch[0].parameters['Stacked Ensemble Classifier']['input_pipelines']
            assert all([pl.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for pl in input_pipelines])


def test_iterative_algorithm_passes_njobs(dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, n_jobs=2, ensembling=False)
    next_batch = algo.next_batch()

    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    for i in range(1, 3):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all([p.parameters['Mock Classifier']['n_jobs'] == 2 for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline, {"id": algo.pipeline_number})


@pytest.mark.parametrize("ensembling_value", [True, False])
def test_iterative_algorithm_one_allowed_pipeline(ensembling_value, logistic_regression_binary_pipeline_class):
    # Checks that when len(allowed_pipeline) == 1, ensembling is not run, even if set to True
    algo = IterativeAlgorithm(allowed_pipelines=[logistic_regression_binary_pipeline_class({})], ensembling=ensembling_value)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == [logistic_regression_binary_pipeline_class({})]

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    assert len(next_batch) == 1
    assert [p.__class__ for p in next_batch] == [logistic_regression_binary_pipeline_class] * len(next_batch)
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert all([p.parameters == p.default_parameters for p in next_batch])
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
        assert [p.__class__ for p in next_batch] == [logistic_regression_binary_pipeline_class] * len(next_batch)
        assert algo.pipeline_number == last_pipeline_number + len(next_batch)
        last_pipeline_number = algo.pipeline_number
        assert algo.batch_number == last_batch_number + 1
        last_batch_number = algo.batch_number
        all_parameters.extend([p.parameters for p in next_batch])
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number})

        assert any([p != logistic_regression_binary_pipeline_class({}).default_parameters for p in all_parameters])


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_binary(n_jobs, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, ensembling=True, n_jobs=n_jobs)
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
                assert pipeline.parameters['Stacked Ensemble Classifier']['n_jobs'] == n_jobs
    assert seen_ensemble


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_regression(n_jobs, linear_regression_pipeline_class):
    algo = IterativeAlgorithm(allowed_pipelines=[linear_regression_pipeline_class({}), linear_regression_pipeline_class({})], ensembling=True, n_jobs=n_jobs)
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
                assert pipeline.parameters['Stacked Ensemble Regressor']['n_jobs'] == n_jobs
    assert seen_ensemble


@pytest.mark.parametrize("parameters", [1, "hello", 1.3, -1.0006, Categorical([1, 3, 4]), Categorical((2, 3, 4))])
def test_iterative_algorithm_pipeline_params(parameters, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes(parameters)
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes,
                              random_seed=0,
                              pipeline_params={'pipeline': {"gap": 2, "max_delay": 10},
                                               'Mock Classifier': {'dummy_parameter': parameters}})

    next_batch = algo.next_batch()
    parameter = parameters
    if isinstance(parameter, Categorical):
        parameter = parameter.rvs(random_state=0)
    assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])
    assert all([p.parameters['Mock Classifier'] == {"dummy_parameter": parameter, "n_jobs": -1} for p in next_batch])

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        for p in next_batch:
            if isinstance(parameters, Categorical):
                assert p.parameters['Mock Classifier']['dummy_parameter'] in parameters
            else:
                assert p.parameters['Mock Classifier']['dummy_parameter'] == parameter


def test_iterative_algorithm_frozen_parameters():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.RANDOM_FOREST
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {'dummy_int_parameter': Integer(1, 10),
                                 'dummy_categorical_parameter': Categorical(["random", "dummy", "test"]),
                                 'dummy_real_parameter': Real(0, 1)}

        def __init__(self, dummy_int_parameter=0, dummy_categorical_parameter='dummy', dummy_real_parameter=1.0, n_jobs=-1, random_seed=0, **kwargs):
            super().__init__(parameters={'dummy_int_parameter': dummy_int_parameter,
                                         'dummy_categorical_parameter': dummy_categorical_parameter,
                                         'dummy_real_parameter': dummy_real_parameter,
                                         **kwargs, 'n_jobs': n_jobs},
                             component_obj=None, random_seed=random_seed)

    pipeline = BinaryClassificationPipeline([MockEstimator])
    algo = IterativeAlgorithm(allowed_pipelines=[pipeline, pipeline, pipeline],
                              pipeline_params={'pipeline': {'date_index': "Date", "gap": 2, "max_delay": 10}},
                              random_seed=0,
                              _frozen_pipeline_parameters={
                                  "Mock Classifier": {
                                      'dummy_int_parameter': 6,
                                      'dummy_categorical_parameter': "random",
                                      'dummy_real_parameter': 0.1
                                  }})

    next_batch = algo.next_batch()
    assert all([p.parameters['pipeline'] == {'date_index': "Date", "gap": 2, "max_delay": 10} for p in next_batch])
    assert all([p.parameters['Mock Classifier'] == {
        'dummy_int_parameter': 6,
        'dummy_categorical_parameter': "random",
        'dummy_real_parameter': 0.1,
        "n_jobs": -1
    } for p in next_batch])

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        assert all([p.parameters['Mock Classifier'] == {
            'dummy_int_parameter': 6,
            'dummy_categorical_parameter': "random",
            'dummy_real_parameter': 0.1,
            "n_jobs": -1
        } for p in next_batch])


def test_iterative_algorithm_pipeline_params_kwargs(dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes,
                              pipeline_params={'Mock Classifier': {'dummy_parameter': "dummy", 'fake_param': 'fake'}},
                              random_seed=0)

    next_batch = algo.next_batch()
    assert all([p.parameters['Mock Classifier'] == {"dummy_parameter": "dummy", "n_jobs": -1, "fake_param": "fake"} for p in next_batch])


def test_iterative_algorithm_results_best_pipeline_info_id(dummy_binary_pipeline_classes, logistic_regression_binary_pipeline_class):
    allowed_pipelines = [dummy_binary_pipeline_classes()[0], logistic_regression_binary_pipeline_class({})]
    algo = IterativeAlgorithm(allowed_pipelines=allowed_pipelines)

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    scores = np.arange(0, len(next_batch))
    for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number + pipeline_num})
    assert algo._best_pipeline_info[ModelFamily.RANDOM_FOREST]['id'] == 3
    assert algo._best_pipeline_info[ModelFamily.LINEAR_MODEL]['id'] == 2

    for i in range(1, 3):
        next_batch = algo.next_batch()
        scores = -np.arange(1, len(next_batch))  # Score always gets better with each pipeline
        for pipeline_num, (score, pipeline) in enumerate(zip(scores, next_batch)):
            algo.add_result(score, pipeline, {"id": algo.pipeline_number + pipeline_num})
            assert algo._best_pipeline_info[pipeline.model_family]['id'] == algo.pipeline_number + pipeline_num


@pytest.mark.parametrize("problem_type", [ProblemTypes.REGRESSION, ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_iterative_algorithm_first_batch_order(problem_type, X_y_binary, has_minimal_dependencies):
    X, y = X_y_binary
    estimators = get_estimators(problem_type, None)
    pipelines = [make_pipeline(X, y, e, problem_type) for e in estimators]
    algo = IterativeAlgorithm(allowed_pipelines=pipelines)

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    estimators_in_first_batch = [p.estimator.name for p in next_batch]

    if problem_type == ProblemTypes.REGRESSION:
        final_estimators = ['XGBoost Regressor',
                            'LightGBM Regressor',
                            'CatBoost Regressor']
    else:
        final_estimators = ['XGBoost Classifier',
                            'LightGBM Classifier',
                            'CatBoost Classifier']
    if has_minimal_dependencies:
        final_estimators = []
    if problem_type == ProblemTypes.REGRESSION:
        assert estimators_in_first_batch == ['Linear Regressor',
                                             'Elastic Net Regressor',
                                             'Decision Tree Regressor',
                                             'Extra Trees Regressor',
                                             'Random Forest Regressor'] + final_estimators
    if problem_type == ProblemTypes.BINARY:
        assert estimators_in_first_batch == ['Elastic Net Classifier',
                                             'Logistic Regression Classifier',
                                             'Decision Tree Classifier',
                                             'Extra Trees Classifier',
                                             'Random Forest Classifier'] + final_estimators
    if problem_type == ProblemTypes.MULTICLASS:
        assert estimators_in_first_batch == ['Elastic Net Classifier',
                                             'Logistic Regression Classifier',
                                             'Decision Tree Classifier',
                                             'Extra Trees Classifier',
                                             'Random Forest Classifier'] + final_estimators


def test_iterative_algorithm_first_batch_order_param(X_y_binary, has_minimal_dependencies):
    X, y = X_y_binary
    estimators = get_estimators('binary', None)
    pipelines = [make_pipeline(X, y, e, 'binary') for e in estimators]
    # put random forest first
    estimator_family_order = [
        ModelFamily.RANDOM_FOREST,
        ModelFamily.LINEAR_MODEL,
        ModelFamily.DECISION_TREE,
        ModelFamily.EXTRA_TREES,
        ModelFamily.XGBOOST,
        ModelFamily.LIGHTGBM,
        ModelFamily.CATBOOST
    ]
    algo = IterativeAlgorithm(allowed_pipelines=pipelines, _estimator_family_order=estimator_family_order)
    next_batch = algo.next_batch()
    estimators_in_first_batch = [p.estimator.name for p in next_batch]

    final_estimators = ['XGBoost Classifier',
                        'LightGBM Classifier',
                        'CatBoost Classifier']
    if has_minimal_dependencies:
        final_estimators = []
    assert estimators_in_first_batch == ['Random Forest Classifier',
                                         'Elastic Net Classifier',
                                         'Logistic Regression Classifier',
                                         'Decision Tree Classifier',
                                         'Extra Trees Classifier'] + final_estimators


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS])
def test_iterative_algorithm_sampling_params(problem_type, mock_imbalanced_data_X_y):
    X, y = mock_imbalanced_data_X_y(problem_type, "some", 'small')
    estimators = get_estimators(problem_type, None)
    pipelines = [make_pipeline(X, y, e, problem_type, sampler_name='Undersampler') for e in estimators]
    algo = IterativeAlgorithm(allowed_pipelines=pipelines,
                              random_seed=0,
                              _frozen_pipeline_parameters={"Undersampler": {"sampling_ratio": 0.5}})

    next_batch = algo.next_batch()
    for p in next_batch:
        for component in p._component_graph:
            if "sampler" in component.name:
                assert component.parameters["sampling_ratio"] == 0.5

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline, {"id": algo.pipeline_number})

    # # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        for p in next_batch:
            for component in p._component_graph:
                if "sampler" in component.name:
                    assert component.parameters["sampling_ratio"] == 0.5
