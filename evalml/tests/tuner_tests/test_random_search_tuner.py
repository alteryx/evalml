from unittest.mock import patch

import numpy as np
import pytest

from evalml import AutoRegressionSearch
from evalml.tuners import NoParamsException, RandomSearchTuner
from evalml.tuners.tuner import Tuner

random_state = 0


def test_random_search_tuner_inheritance():
    assert issubclass(RandomSearchTuner, Tuner)


def test_random_search_tuner_automl(X_y):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=RandomSearchTuner)
    clf.search(X, y)


def test_random_search_tuner_automl_no_params(X_y, capsys):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=20, allowed_model_families=['linear_model'], tuner=RandomSearchTuner)
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


@patch('evalml.tuners.RandomSearchTuner.is_search_space_exhausted')
def test_random_search_tuner_exhausted_space(mock_is_search_space_exhausted, X_y):
    mock_is_search_space_exhausted.side_effects = lambda: False
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=RandomSearchTuner)
    clf.search(X, y)
    assert len(clf.results['pipeline_results']) == 0


def test_random_search_tuner_unique_values(dummy_component_hyperparameters, dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters)
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state)
    generated_parameters = []
    for i in range(10):
        params = tuner.propose()
        generated_parameters.append(params)
    assert len(generated_parameters) == 10
    for i in range(10):
        assert generated_parameters[i].keys() == {'Mock Classifier'}
        assert generated_parameters[i]['Mock Classifier'].keys() == dummy_component_hyperparameters.keys()


def test_random_search_tuner_no_params(dummy_component_hyperparameters_small, dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters_small)
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state, with_replacement=False)
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            tuner.propose()


def test_random_search_tuner_basic(dummy_component_hyperparameters,
                                   dummy_component_hyperparameters_unicode,
                                   dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters)
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'column a': 5,
            'column b': 8.442657485810175,
            'column c': 'option c',
            'column d': np.inf
        }
    }
    tuner.add(proposed_params, 0.5)

    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters_unicode)
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'column a': 5,
            'column b': 8.442657485810175,
            'column c': 'option c 💩',
            'column d': np.inf
        }
    }
    tuner.add(proposed_params, 0.5)


def test_random_search_tuner_space_types(dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({'column a': (0, 10)})
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'column a': 5}}

    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({'column a': (0, 10.0)})
    tuner = RandomSearchTuner(MockBinaryClassificationPipeline, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'column a': 5.488135039273248}}


def test_random_search_tuner_invalid_space(dummy_binary_pipeline_class):
    value_error_text = 'Dimension has to be a list or tuple'
    bound_error_text = "has to be less than the upper bound"
    with pytest.raises(ValueError, match=value_error_text):
        RandomSearchTuner(dummy_binary_pipeline_class({'column a': False}), random_state=random_state)
    with pytest.raises(ValueError, match=value_error_text):
        RandomSearchTuner(dummy_binary_pipeline_class({'column a': (0)}), random_state=random_state)
    with pytest.raises(ValueError, match=bound_error_text):
        RandomSearchTuner(dummy_binary_pipeline_class({'column a': (1, 0)}), random_state=random_state)
    with pytest.raises(ValueError, match=bound_error_text):
        RandomSearchTuner(dummy_binary_pipeline_class({'column a': (0, 0)}), random_state=random_state)
