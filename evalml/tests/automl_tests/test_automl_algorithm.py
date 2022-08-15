import pytest

from evalml.automl.automl_algorithm import AutoMLAlgorithm
from evalml.exceptions import PipelineNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import (
    DecisionTreeClassifier,
    LogisticRegressionClassifier,
)


class DummyAlgorithm(AutoMLAlgorithm):
    def __init__(self, dummy_pipelines=None):
        super().__init__()
        self._dummy_pipelines = dummy_pipelines or []

    def num_pipelines_per_batch(self, batch_number):
        return None

    def next_batch(self):
        self._pipeline_number += 1
        self._batch_number += 1
        if len(self._dummy_pipelines) > 0:
            return self._dummy_pipelines.pop()
        raise StopIteration("No more pipelines!")

    def _transform_parameters(self, pipeline, proposed_parameters):
        pass


class AllowedPipelinesAlgorithm(AutoMLAlgorithm):
    def __init__(self, allowed_pipelines=None, random_seed=0):
        super().__init__(allowed_pipelines=allowed_pipelines, random_seed=random_seed)

    def num_pipelines_per_batch(self, batch_number):
        return None

    def next_batch(self):
        pass

    def _transform_parameters(self, pipeline, proposed_parameters):
        pass


def test_automl_algorithm_dummy():
    algo = DummyAlgorithm()
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0

    algo = DummyAlgorithm(dummy_pipelines=["pipeline 3", "pipeline 2", "pipeline 1"])
    assert algo.pipeline_number == 0
    assert algo.batch_number == 0
    assert algo.next_batch() == "pipeline 1"
    assert algo.pipeline_number == 1
    assert algo.batch_number == 1
    assert algo.next_batch() == "pipeline 2"
    assert algo.pipeline_number == 2
    assert algo.batch_number == 2
    assert algo.next_batch() == "pipeline 3"
    assert algo.pipeline_number == 3
    assert algo.batch_number == 3
    assert algo.num_pipelines_per_batch(3) is None
    with pytest.raises(StopIteration, match="No more pipelines!"):
        algo.next_batch()


def test_automl_algorithm_invalid_pipeline_add(dummy_regression_pipeline):
    algo = DummyAlgorithm()
    with pytest.raises(
        PipelineNotFoundError,
        match="No such pipeline allowed in this AutoML search: Mock Regression Pipeline",
    ):
        algo.add_result(0.1234, dummy_regression_pipeline, {})


def test_automl_algorithm_create_ensemble_cache():
    algo = DummyAlgorithm()
    lrc = LogisticRegressionClassifier()
    dtc = DecisionTreeClassifier()
    bcp_linear = BinaryClassificationPipeline([lrc])
    bcp_trees = BinaryClassificationPipeline([dtc])
    bpi = {
        ModelFamily.LINEAR_MODEL: {
            "mean_cv_score": 0.5,
            "pipeline": bcp_linear,
            "parameters": bcp_linear.parameters,
            "id": 1,
            "cached_data": {"hash1": {"Logistic Regression Classifier": lrc}},
        },
        ModelFamily.DECISION_TREE: {
            "mean_cv_score": 0.5,
            "pipeline": bcp_trees,
            "parameters": bcp_trees.parameters,
            "id": 1,
            "cached_data": {"hash1": {"Decision Tree Classifier": dtc}},
        },
    }
    algo._best_pipeline_info = bpi
    pipelines = algo._create_ensemble()[0]

    # check component graph expected cache
    expected_comp_graph = {
        "hash1": {
            "Linear Pipeline - Logistic Regression Classifier": lrc,
            "Decision Tree Pipeline - Decision Tree Classifier": dtc,
        },
    }
    assert pipelines.component_graph.cached_data == expected_comp_graph


def test_automl_algorithm_add_pipelines(dummy_binary_pipeline):
    allowed_pipelines = [dummy_binary_pipeline]
    aml = AllowedPipelinesAlgorithm(allowed_pipelines=allowed_pipelines)
    aml_add_pipelines = AllowedPipelinesAlgorithm()
    aml_add_pipelines._set_allowed_pipelines(allowed_pipelines)

    assert aml.allowed_pipelines == aml_add_pipelines.allowed_pipelines
    assert aml.num_pipelines_per_batch(0) is None
    # the tuner objects themselves are different so we cannot check for dictionary equality
    assert aml._tuners.keys() == aml_add_pipelines._tuners.keys()
    assert aml._tuner_class == aml_add_pipelines._tuner_class
    aml.next_batch()
    aml._transform_parameters(None, None)
