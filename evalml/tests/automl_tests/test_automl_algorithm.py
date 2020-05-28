import pytest

from evalml.automl.automl_algorithm import AutoMLAlgorithm
from evalml.objectives import F1


def test_automl_algorithm_init_base():
    with pytest.raises(TypeError):
        AutoMLAlgorithm()
    with pytest.raises(TypeError):
        AutoMLAlgorithm(F1)


class DummyAlgorithm(AutoMLAlgorithm):
    def __init__(self, objective, dummy_pipelines=None):
        super().__init__(objective)
        self._dummy_pipelines = dummy_pipelines or []

    def can_continue(self):
        return len(self._dummy_pipelines) > 0

    def next_batch(self):
        self._pipeline_number += 1
        self._batch_number += 1
        if len(self._dummy_pipelines) > 0:
            return self._dummy_pipelines.pop()
        return None


def test_automl_algorithm_dummy():
    algo = DummyAlgorithm(F1)
    assert not algo.can_continue()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0

    algo = DummyAlgorithm(F1, dummy_pipelines=['pipeline 3', 'pipeline 2', 'pipeline 1'])
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.can_continue()
    assert algo.next_batch() == 'pipeline 1'
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert algo.can_continue()
    assert algo.next_batch() == 'pipeline 2'
    assert algo.pipeline_number == 2
    assert algo.batch_number == 2
    assert algo.can_continue()
    assert algo.next_batch() == 'pipeline 3'
    assert algo.pipeline_number == 3
    assert algo.batch_number == 3
    assert not algo.can_continue()
    assert algo.next_batch() is None
