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
from evalml.pipelines.components.transformers import TextFeaturizer
from evalml.problem_types import ProblemTypes


def test_iterative_algorithm_init_iterative():
    IterativeAlgorithm()


def test_iterative_algorithm_init():
    algo = IterativeAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []


def test_iterative_algorithm_allowed_pipelines(logistic_regression_binary_pipeline_class):
    allowed_pipelines = [logistic_regression_binary_pipeline_class]
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

            def __init__(self, dummy_parameter='default', n_jobs=-1, random_state=0, **kwargs):
                super().__init__(parameters={'dummy_parameter': dummy_parameter, **kwargs,
                                             'n_jobs': n_jobs},
                                 component_obj=None, random_state=random_state)

        class MockBinaryClassificationPipeline1(BinaryClassificationPipeline):
            estimator = MockEstimator
            component_graph = [MockEstimator]

        class MockBinaryClassificationPipeline2(BinaryClassificationPipeline):
            estimator = MockEstimator
            component_graph = [MockEstimator]

        class MockBinaryClassificationPipeline3(BinaryClassificationPipeline):
            estimator = MockEstimator
            component_graph = [MockEstimator]

        return [MockBinaryClassificationPipeline1,
                MockBinaryClassificationPipeline2,
                MockBinaryClassificationPipeline3]
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
    assert [p.__class__ for p in next_batch] == dummy_binary_pipeline_classes
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert all([p.parameters == p.__class__.default_parameters for p in next_batch])
    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)

    # subsequent batches contain pipelines_per_batch copies of one pipeline, moving from best to worst from the first batch
    last_batch_number = algo.batch_number
    last_pipeline_number = algo.pipeline_number
    all_parameters = []

    for i in range(1, 5):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert len(next_batch) == algo.pipelines_per_batch
            num_pipelines_classes = (len(dummy_binary_pipeline_classes) + 1) if ensembling_value else len(dummy_binary_pipeline_classes)
            cls = dummy_binary_pipeline_classes[(algo.batch_number - 2) % num_pipelines_classes]
            assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
            assert all([p.parameters['Mock Classifier']['n_jobs'] == -1 for p in next_batch])
            assert all((p.random_state == algo.random_state) for p in next_batch)
            assert algo.pipeline_number == last_pipeline_number + len(next_batch)
            last_pipeline_number = algo.pipeline_number
            assert algo.batch_number == last_batch_number + 1
            last_batch_number = algo.batch_number
            all_parameters.extend([p.parameters for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline)
        assert any([p != dummy_binary_pipeline_classes[0]({}).parameters for p in all_parameters])

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
                algo.add_result(score, pipeline)
            assert pipeline.model_family == ModelFamily.ENSEMBLE
            assert pipeline.random_state == algo.random_state
            stack_args = mock_stack.call_args[1]['estimators']
            estimators_used_in_ensemble = [args[1] for args in stack_args]
            random_states_the_same = [(estimator.pipeline.random_state == algo.random_state)
                                      for estimator in estimators_used_in_ensemble]
            assert all(random_states_the_same)


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
        algo.add_result(score, pipeline)

    for i in range(1, 5):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline)

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
        algo.add_result(score, pipeline)

    for i in range(1, 3):
        for _ in range(len(dummy_binary_pipeline_classes)):
            next_batch = algo.next_batch()
            assert all([p.parameters['Mock Classifier']['n_jobs'] == 2 for p in next_batch])
            scores = -np.arange(0, len(next_batch))
            for score, pipeline in zip(scores, next_batch):
                algo.add_result(score, pipeline)


@pytest.mark.parametrize("ensembling_value", [True, False])
def test_iterative_algorithm_one_allowed_pipeline(ensembling_value, logistic_regression_binary_pipeline_class):
    # Checks that when len(allowed_pipeline) == 1, ensembling is not run, even if set to True
    algo = IterativeAlgorithm(allowed_pipelines=[logistic_regression_binary_pipeline_class], ensembling=ensembling_value)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == [logistic_regression_binary_pipeline_class]

    # initial batch contains one of each pipeline, with default parameters
    next_batch = algo.next_batch()
    assert len(next_batch) == 1
    assert [p.__class__ for p in next_batch] == [logistic_regression_binary_pipeline_class] * len(next_batch)
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert all([p.parameters == p.__class__.default_parameters for p in next_batch])
    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)

    # subsequent batches contain pipelines_per_batch copies of one pipeline, moving from best to worst from the first batch
    last_batch_number = algo.batch_number
    last_pipeline_number = algo.pipeline_number
    all_parameters = []
    for i in range(1, 5):
        next_batch = algo.next_batch()
        assert len(next_batch) == algo.pipelines_per_batch
        assert all((p.random_state == algo.random_state) for p in next_batch)
        assert [p.__class__ for p in next_batch] == [logistic_regression_binary_pipeline_class] * len(next_batch)
        assert algo.pipeline_number == last_pipeline_number + len(next_batch)
        last_pipeline_number = algo.pipeline_number
        assert algo.batch_number == last_batch_number + 1
        last_batch_number = algo.batch_number
        all_parameters.extend([p.parameters for p in next_batch])
        scores = -np.arange(0, len(next_batch))
        for score, pipeline in zip(scores, next_batch):
            algo.add_result(score, pipeline)
        assert any([p != logistic_regression_binary_pipeline_class.default_parameters for p in all_parameters])


def test_iterative_algorithm_instantiates_text(dummy_classifier_estimator_class):
    class MockTextClassificationPipeline(BinaryClassificationPipeline):
        component_graph = [TextFeaturizer, dummy_classifier_estimator_class]

    algo = IterativeAlgorithm(allowed_pipelines=[MockTextClassificationPipeline], text_columns=['text_col_1', 'text_col_2'])
    pipeline = algo.next_batch()[0]
    expected_params = {'text_columns': ['text_col_1', 'text_col_2']}
    assert pipeline.parameters['Text Featurization Component'] == expected_params
    assert isinstance(pipeline[0], TextFeaturizer)
    assert pipeline[0]._all_text_columns == ['text_col_1', 'text_col_2']


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_binary(n_jobs, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, ensembling=True, n_jobs=n_jobs)
    next_batch = algo.next_batch()
    seen_ensemble = False
    scores = range(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)
    for i in range(5):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            if isinstance(pipeline.estimator, StackedEnsembleClassifier):
                seen_ensemble = True
                assert pipeline.parameters['Stacked Ensemble Classifier']['n_jobs'] == n_jobs
    assert seen_ensemble


@pytest.mark.parametrize("n_jobs", [-1, 0, 1, 2, 3])
def test_iterative_algorithm_stacked_ensemble_n_jobs_regression(n_jobs, linear_regression_pipeline_class):
    algo = IterativeAlgorithm(allowed_pipelines=[linear_regression_pipeline_class, linear_regression_pipeline_class], ensembling=True, n_jobs=n_jobs)
    next_batch = algo.next_batch()
    seen_ensemble = False
    scores = range(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)
    for i in range(5):
        next_batch = algo.next_batch()
        for pipeline in next_batch:
            if isinstance(pipeline.estimator, StackedEnsembleRegressor):
                seen_ensemble = True
                assert pipeline.parameters['Stacked Ensemble Regressor']['n_jobs'] == n_jobs
    assert seen_ensemble


@pytest.mark.parametrize("parameters", [1, "hello", 1.3, -1.0006, [1, 3, 4], (2, 3, 4)])
def test_iterative_algorithm_pipeline_params(parameters, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes(parameters)
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes,
                              pipeline_params={'pipeline': {"gap": 2, "max_delay": 10},
                                               'Mock Classifier': {'dummy_parameter': parameters}})

    next_batch = algo.next_batch()
    parameter = parameters
    if isinstance(parameter, (list, tuple)):
        parameter = parameters[0]
    assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])
    assert all([p.parameters['Mock Classifier'] == {"dummy_parameter": parameter, "n_jobs": -1} for p in next_batch])

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)

    # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        for p in next_batch:
            if isinstance(parameters, (tuple, list)):
                assert p.parameters['Mock Classifier']['dummy_parameter'] in parameters
            else:
                assert p.parameters['Mock Classifier']['dummy_parameter'] == parameter


@pytest.mark.parametrize("parameters", [Real(0, 1), Categorical(["random", "dummy", "test"]), Integer(1, 10)])
def test_iterative_algorithm_pipeline_params_skopt(parameters, dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes(parameters)
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes,
                              pipeline_params={'pipeline': {"gap": 2, "max_delay": 10},
                                               'Mock Classifier': {'dummy_parameter': parameters}},
                              random_state=0)

    next_batch = algo.next_batch()
    if isinstance(parameters, (Real, Integer)):
        parameter = parameters.rvs(random_state=0)[0]
    else:
        parameter = parameters.rvs(random_state=0)
    assert all([p.parameters['pipeline'] == {"gap": 2, "max_delay": 10} for p in next_batch])
    assert all([p.parameters['Mock Classifier'] == {"dummy_parameter": parameter, "n_jobs": -1} for p in next_batch])

    scores = np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)

    # make sure that future batches remain in the hyperparam range
    for i in range(1, 5):
        next_batch = algo.next_batch()
        for p in next_batch:
            if isinstance(parameters, Categorical):
                assert p.parameters['Mock Classifier']['dummy_parameter'] in ["random", "dummy", "test"]
            elif isinstance(parameters, Real):
                assert 0 < p.parameters['Mock Classifier']['dummy_parameter'] <= 1
            else:
                assert 1 <= p.parameters['Mock Classifier']['dummy_parameter'] <= 10


def test_iterative_algorithm_pipeline_params_kwargs(dummy_binary_pipeline_classes):
    dummy_binary_pipeline_classes = dummy_binary_pipeline_classes()
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes,
                              pipeline_params={'Mock Classifier': {'dummy_parameter': "dummy", 'fake_param': 'fake'}},
                              random_state=0)

    next_batch = algo.next_batch()
    assert all([p.parameters['Mock Classifier'] == {"dummy_parameter": "dummy", "n_jobs": -1, "fake_param": "fake"} for p in next_batch])
