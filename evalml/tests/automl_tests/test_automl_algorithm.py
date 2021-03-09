import pytest

from evalml.automl.automl_algorithm import AutoMLAlgorithm
from evalml.exceptions import PipelineNotFoundError


class DummyAlgorithm(AutoMLAlgorithm):
    def __init__(self, dummy_pipelines=None):
        super().__init__()
        self._dummy_pipelines = dummy_pipelines or []

    def next_batch(self):
        self._pipeline_number += 1
        self._batch_number += 1
        if len(self._dummy_pipelines) > 0:
            return self._dummy_pipelines.pop()
        raise StopIteration('No more pipelines!')


def test_automl_algorithm_dummy():
    algo = DummyAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0

    algo = DummyAlgorithm(dummy_pipelines=['pipeline 3', 'pipeline 2', 'pipeline 1'])
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.next_batch() == 'pipeline 1'
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert algo.next_batch() == 'pipeline 2'
    assert algo.pipeline_number == 2
    assert algo.batch_number == 2
    assert algo.next_batch() == 'pipeline 3'
    assert algo.pipeline_number == 3
    assert algo.batch_number == 3
    with pytest.raises(StopIteration, match='No more pipelines!'):
        algo.next_batch()


def test_automl_algorithm_invalid_pipeline_add(dummy_regression_pipeline_class):
    algo = DummyAlgorithm()
    pipeline = dummy_regression_pipeline_class(parameters={})
    with pytest.raises(PipelineNotFoundError, match="No such pipeline allowed in this AutoML search: Mock Regression Pipeline"):
        algo.add_result(0.1234, pipeline, {})
