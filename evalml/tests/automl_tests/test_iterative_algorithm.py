from unittest.mock import patch

import numpy as np
import pytest

from evalml.automl.automl_algorithm import (
    AutoMLAlgorithmException,
    IterativeAlgorithm
)
from evalml.objectives import F1, LogLossBinary
from evalml.pipelines import BinaryClassificationPipeline, get_pipelines
from evalml.problem_types import ProblemTypes


def test_iterative_algorithm_init_iterative():
    IterativeAlgorithm(F1)


def test_iterative_algorithm_init():
    algo = IterativeAlgorithm(F1)
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.can_continue()
    assert algo.allowed_pipelines == get_pipelines(problem_type=ProblemTypes.BINARY)


@pytest.fixture
def dummy_binary_pipeline_classes(dummy_classifier_estimator_class):
    MockEstimator = dummy_classifier_estimator_class

    class MockBinaryClassificationPipeline1(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    class MockBinaryClassificationPipeline2(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    class MockBinaryClassificationPipeline3(BinaryClassificationPipeline):
        estimator = MockEstimator
        component_graph = [MockEstimator()]

    return [MockBinaryClassificationPipeline1,
            MockBinaryClassificationPipeline2,
            MockBinaryClassificationPipeline3]


@patch('evalml.automl.automl_algorithm.automl_algorithm.get_pipelines')
def test_iterative_algorithm_empty(mock_get_pipelines, dummy_binary_pipeline_classes):
    mock_get_pipelines.return_value = dummy_binary_pipeline_classes
    algo = IterativeAlgorithm(LogLossBinary)
    mock_get_pipelines.assert_called_once()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.can_continue()

    next_batch = algo.next_batch()
    assert [p.__class__ for p in next_batch] == dummy_binary_pipeline_classes
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert algo.can_continue()

    with pytest.raises(AutoMLAlgorithmException, match='Some results are needed before the next automl batch can be computed.'):
        algo.next_batch()
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert algo.can_continue()


@patch('evalml.automl.automl_algorithm.automl_algorithm.get_pipelines')
def test_iterative_algorithm_results(mock_get_pipelines, dummy_binary_pipeline_classes):
    mock_get_pipelines.return_value = dummy_binary_pipeline_classes
    algo = IterativeAlgorithm(LogLossBinary)
    mock_get_pipelines.assert_called_once()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.can_continue()

    # initial batch contains one of each pipeline
    next_batch = algo.next_batch()
    assert len(next_batch) == len(dummy_binary_pipeline_classes)
    assert [p.__class__ for p in next_batch] == dummy_binary_pipeline_classes
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes)
    assert algo.batch_number == 1
    assert algo.can_continue()
    # the "best" score will be the 1st dummy pipeline
    scores = np.arange(0, len(next_batch))
    pipelines = [cls({}) for cls in dummy_binary_pipeline_classes]
    for score, pipeline in zip(scores, pipelines):
        algo.add_result(score, pipeline)

    # subsequent batches contain samples_per_batch copies of one pipeline, moving from best to worst from the first batch
    next_batch = algo.next_batch()
    assert len(next_batch) == algo.samples_per_batch
    cls = dummy_binary_pipeline_classes[0]
    assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes) + algo.samples_per_batch
    assert algo.batch_number == 2
    assert algo.can_continue()
    scores = -np.arange(0, len(next_batch))
    pipelines = [cls({}) for pipeline in next_batch]
    for score, pipeline in zip(scores, pipelines):
        algo.add_result(score, pipeline)

    next_batch = algo.next_batch()
    assert len(next_batch) == algo.samples_per_batch
    cls = dummy_binary_pipeline_classes[1]
    assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes) + 2 * algo.samples_per_batch
    assert algo.batch_number == 3
    assert algo.can_continue()
    scores = -np.arange(0, len(next_batch))
    pipelines = [cls({}) for pipeline in next_batch]
    for score, pipeline in zip(scores, pipelines):
        algo.add_result(score, pipeline)

    next_batch = algo.next_batch()
    assert len(next_batch) == algo.samples_per_batch
    cls = dummy_binary_pipeline_classes[2]
    assert [p.__class__ for p in next_batch] == [cls] * len(next_batch)
    assert algo.pipeline_number == len(dummy_binary_pipeline_classes) + 3 * algo.samples_per_batch
    assert algo.batch_number == 4
    assert not algo.can_continue()
