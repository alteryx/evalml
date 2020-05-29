import numpy as np
import pytest

from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.model_family import ModelFamily
from evalml.pipelines import (
    BinaryClassificationPipeline,
    LogisticRegressionBinaryPipeline
)
from evalml.pipelines.components import Estimator
from evalml.problem_types import ProblemTypes


def test_iterative_algorithm_init_iterative():
    IterativeAlgorithm()


def test_iterative_algorithm_init():
    algo = IterativeAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []
    assert algo.allowed_model_families == []


def test_iterative_algorithm_allowed_inputs():
    allowed_pipelines = [LogisticRegressionBinaryPipeline]
    allowed_model_families = [ModelFamily.RANDOM_FOREST]
    algo = IterativeAlgorithm(allowed_pipelines=allowed_pipelines, allowed_model_families=allowed_model_families)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_model_families == allowed_model_families
    assert algo.allowed_pipelines == allowed_pipelines


@pytest.fixture
def dummy_binary_pipeline_classes():
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = {'dummy_parameter': ['default', 'other']}

        def __init__(self, dummy_parameter='default', random_state=0):
            super().__init__(parameters={'dummy_parameter': dummy_parameter}, component_obj=None, random_state=random_state)

    class MockBinaryClassificationPipeline1(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    class MockBinaryClassificationPipeline2(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    return [MockBinaryClassificationPipeline1,
            MockBinaryClassificationPipeline2]


def test_iterative_algorithm_empty(dummy_binary_pipeline_classes):
    algo = IterativeAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == []
    assert algo.allowed_model_families == []

    next_batch = algo.next_batch()
    assert [p.__class__ for p in next_batch] == []
    assert algo.pipeline_number == 0
    assert algo.batch_number == 1

    with pytest.raises(StopIteration):
        assert algo.next_batch() == []
    assert algo.batch_number == 1
    assert algo.pipeline_number == 0


def test_iterative_algorithm_results(dummy_binary_pipeline_classes):
    algo = IterativeAlgorithm(allowed_pipelines=dummy_binary_pipeline_classes, allowed_model_families=[ModelFamily.NONE])
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.allowed_pipelines == dummy_binary_pipeline_classes
    assert algo.allowed_model_families == [ModelFamily.NONE]

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
    next_batch = algo.next_batch()
    assert len(next_batch) == algo.pipelines_per_batch
    cls = dummy_binary_pipeline_classes[0]
    assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes) + algo.pipelines_per_batch
    assert algo.batch_number == 2
    print([p.parameters for p in next_batch])
    assert any([p.parameters != (p.__class__)({}).parameters for p in next_batch])
    scores = -np.arange(0, len(next_batch))
    for score, pipeline in zip(scores, next_batch):
        algo.add_result(score, pipeline)

    next_batch = algo.next_batch()
    assert len(next_batch) == algo.pipelines_per_batch
    cls = dummy_binary_pipeline_classes[1]
    assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes) + 2 * algo.pipelines_per_batch
    assert algo.batch_number == 3
    assert any([p.parameters != (p.__class__)({}).parameters for p in next_batch])

    with pytest.raises(StopIteration, match='No more batches available'):
        algo.next_batch()
