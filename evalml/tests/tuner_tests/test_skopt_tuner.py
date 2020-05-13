#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pytest
from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.components import Estimator
from evalml.problem_types import ProblemTypes
from evalml.tuners.skopt_tuner import SKOptTuner
from evalml.tuners.tuner import Tuner

random_state = 0


def test_tuner_base(dummy_binary_pipeline):
    with pytest.raises(TypeError, match="'MockBinaryClassificationPipeline' object is not callable"):
        Tuner(dummy_binary_pipeline({}))
    with pytest.raises(TypeError, match="Can't instantiate abstract class Tuner with abstract methods add, propose"):
        Tuner(dummy_binary_pipeline)


def dummy_estimator_class(_hyperparameter_ranges):
    class MockEstimator(Estimator):
        name = "Mock Classifier"
        model_family = ModelFamily.NONE
        supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
        hyperparameter_ranges = _hyperparameter_ranges or {}

        def __init__(self, random_state=0):
            super().__init__(parameters={}, component_obj=None, random_state=random_state)
    return MockEstimator


def dummy_binary_pipeline_class(dummy_estimator):
    class MockBinaryClassificationPipeline(BinaryClassificationPipeline):
        estimator = dummy_estimator
        component_graph = [dummy_estimator()]
    return MockBinaryClassificationPipeline


def test_skopt_tuner_init(dummy_binary_pipeline):
    with pytest.raises(TypeError, match='Argument "pipeline_class" must be a class which subclasses PipelineBase'):
        SKOptTuner(dummy_binary_pipeline)

    class X:
        pass

    with pytest.raises(TypeError, match='Argument "pipeline_class" must be a class which subclasses PipelineBase'):
        SKOptTuner(X)
    SKOptTuner(dummy_binary_pipeline_class(dummy_estimator_class({})))


def test_skopt_tuner_basic():
    estimator_hyperparameter_ranges = {
        'parameter a': Integer(0, 10),
        'parameter b': Real(0, 10),
        'parameter c': (0, 10),
        'parameter d': (0, 10.0),
        'parameter e': ['option a', 'option b', 'option c'],
        'parameter f': ['option a 💩', 'option b 💩', 'option c 💩'],
        'parameter g': ['option a', 'option b', 100, np.inf]
    }
    MockEstimator = dummy_estimator_class(estimator_hyperparameter_ranges)
    MockPipeline = dummy_binary_pipeline_class(MockEstimator)

    tuner = SKOptTuner(MockPipeline, random_state=random_state)
    assert isinstance(tuner, Tuner)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'parameter a': 5,
            'parameter b': 8.442657485810175,
            'parameter c': 3,
            'parameter d': 8.472517387841256,
            'parameter e': 'option b',
            'parameter f': 'option b 💩',
            'parameter g': 'option b'
        }
    }
    tuner.add(proposed_params, 0.5)


def test_skopt_tuner_invalid_parameters_score(test_space):
    tuner = SKOptTuner(test_space)
    with pytest.raises(TypeError):
        tuner.add(0, 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, 2, 3], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, '2', '3'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([-1, 1, 'option_a', 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, -1, 'option_a', 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([0, 1, 'option_a', 3, 4], 0.5)
    with pytest.raises(ValueError):
        tuner.add([np.nan, 1, 'option_a', 'option_a'], 0.5)
    with pytest.raises(ValueError):
        tuner.add([np.inf, 1, 'option_a', 'option_a'], 0.5)
    with pytest.raises(TypeError):
        tuner.add([None, 1, 'option_a', 'option_a'], 0.5)
    tuner.add((0, 1, 'option_a', 'option_b'), 0.5)
    tuner.add([0, 1, 'option_a', 100], np.nan)
    tuner.add([0, 1, 'option_a', np.inf], np.inf)
    tuner.add([0, 1, 'option_a', 'option_a'], None)
    tuner.propose()
    print(random_state)
