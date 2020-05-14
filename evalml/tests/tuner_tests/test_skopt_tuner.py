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


def test_tuner_init(dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({})
    with pytest.raises(TypeError, match="Can't instantiate abstract class Tuner with abstract methods add, propose"):
        Tuner(MockBinaryClassificationPipeline({}))
    with pytest.raises(TypeError, match="Can't instantiate abstract class Tuner with abstract methods add, propose"):
        Tuner(MockBinaryClassificationPipeline)


def test_skopt_tuner_init(dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({})

    with pytest.raises(TypeError, match='Argument "pipeline_class" must be a class which subclasses PipelineBase'):
        SKOptTuner(MockBinaryClassificationPipeline({}))

    class X:
        pass

    with pytest.raises(TypeError, match='Argument "pipeline_class" must be a class which subclasses PipelineBase'):
        SKOptTuner(X)
    SKOptTuner(MockBinaryClassificationPipeline)


def test_skopt_tuner_basic(dummy_binary_pipeline_class):
    estimator_hyperparameter_ranges = {
        'parameter a': Integer(0, 10),
        'parameter b': Real(0, 10),
        'parameter c': (0, 10),
        'parameter d': (0, 10.0),
        'parameter e': ['option a', 'option b', 'option c'],
        'parameter f': ['option a 💩', 'option b 💩', 'option c 💩'],
        'parameter g': ['option a', 'option b', 100, np.inf]
    }
    MockPipeline = dummy_binary_pipeline_class(estimator_hyperparameter_ranges)

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


def test_skopt_tuner_invalid_ranges(dummy_binary_pipeline_class):
    tuner = SKOptTuner(dummy_binary_pipeline_class({
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a', 'option b', 'option c']
    }), random_state=random_state)

    with pytest.raises(ValueError, match="Invalid dimension \[\]. Read the documentation for supported types."):
        tuner = SKOptTuner(dummy_binary_pipeline_class({
            'param a': Integer(0, 10),
            'param b': Real(0, 10),
            'param c': []
        }), random_state=random_state)
    with pytest.raises(ValueError, match="Invalid dimension None."):
        tuner = SKOptTuner(dummy_binary_pipeline_class({
            'param a': Integer(0, 10),
            'param b': Real(0, 10),
            'param c': None
        }), random_state=random_state)
    with pytest.raises(ValueError, match="Dimension has to be a list or tuple."):
        tuner = SKOptTuner(dummy_binary_pipeline_class({
            'param a': Integer(0, 10),
            'param b': Real(0, 10),
            'param c': 'Value'
        }), random_state=random_state)


def test_skopt_tuner_invalid_parameters_score(dummy_binary_pipeline_class):
    estimator_hyperparameter_ranges = {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a', 'option b', 'option c']
    }
    MockPipeline = dummy_binary_pipeline_class(estimator_hyperparameter_ranges)

    tuner = SKOptTuner(MockPipeline, random_state=random_state)
    with pytest.raises(TypeError):
        tuner.add({}, 0.5)
    with pytest.raises(TypeError):
        tuner.add({'Mock Classifier': {}}, 0.5)
    with pytest.raises(TypeError):
        tuner.add({'Mock Classifier': {'param a': 0}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': 0, 'param b': 0.0, 'param c': 0}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': -1, 'param b': 0.0, 'param c': 'option a'}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': 0, 'param b': 11.0, 'param c': 'option a'}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': 0, 'param b': 0.0, 'param c': 'option d'}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': np.nan, 'param b': 0.0, 'param c': 'option a'}}, 0.5)
    with pytest.raises(ValueError, match="is not within the bounds of the space"):
        tuner.add({'Mock Classifier': {'param a': np.inf, 'param b': 0.0, 'param c': 'option a'}}, 0.5)
    with pytest.raises(TypeError):
        tuner.add({'Mock Classifier': {'param a': None, 'param b': 0.0, 'param c': 'option a'}}, 0.5)
    tuner.add({'Mock Classifier': {'param a': 0, 'param b': 1.0, 'param c': 'option a'}}, 0.5)
    tuner.add({'Mock Classifier': {'param a': 0, 'param b': 1.0, 'param c': 'option a'}}, np.nan)
    tuner.add({'Mock Classifier': {'param a': 0, 'param b': 1.0, 'param c': 'option a'}}, np.inf)
    tuner.add({'Mock Classifier': {'param a': 0, 'param b': 1.0, 'param c': 'option a'}}, None)
    tuner.propose()
    print(random_state)


def test_skopt_tuner_propose(dummy_binary_pipeline_class):
    estimator_hyperparameter_ranges = {
        'param a': Integer(0, 10),
        'param b': Real(0, 10),
        'param c': ['option a', 'option b', 'option c']
    }
    MockPipeline = dummy_binary_pipeline_class(estimator_hyperparameter_ranges)

    tuner = SKOptTuner(MockPipeline, random_state=random_state)
    tuner.add({'Mock Classifier': {'param a': 0, 'param b': 1.0, 'param c': 'option a'}}, 0.5)
    parameters = tuner.propose()
    assert parameters == {
        'Mock Classifier': {
            'param a': 5,
            'param b': 8.442657485810175,
            'param c': 'option c'
        }
    }
    print(random_state)
