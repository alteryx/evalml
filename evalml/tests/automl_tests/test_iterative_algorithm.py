import numpy as np
import pytest

from evalml.automl.automl_algorithm import (
    AutoMLAlgorithmException,
    IterativeAlgorithm
)
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import Estimator
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
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.RANDOM_FOREST
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {'dummy_parameter': ['default', 'other']}

        def __init__(self, dummy_parameter='default', random_state=0):
            super().__init__(parameters={'dummy_parameter': dummy_parameter}, component_obj=None, random_state=random_state)

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
def test_iterative_algorithm_results(ensembling_value, dummy_binary_pipeline_classes):
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
    assert all([p.parameters == (p.__class__)({}).parameters for p in next_batch])
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
            assert algo.pipeline_number == last_pipeline_number + len(next_batch)
            last_pipeline_number = algo.pipeline_number
            assert algo.batch_number == last_batch_number + 1
            last_batch_number = algo.batch_number
            print([p.parameters for p in next_batch])
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
