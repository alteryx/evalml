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
    assert len(clf.results['pipeline_results']) == 1   # baseline pipeline


def test_random_search_tuner_unique_values(dummy_pipeline_hyperparameters):
    tuner = RandomSearchTuner(dummy_pipeline_hyperparameters, random_state=random_state)
    generated_parameters = []
    for i in range(10):
        params = tuner.propose()
        generated_parameters.append(params)
    assert len(generated_parameters) == 10
    for i in range(10):
        assert generated_parameters[i].keys() == dummy_pipeline_hyperparameters.keys()
        assert generated_parameters[i]['Mock Classifier'].keys() == dummy_pipeline_hyperparameters['Mock Classifier'].keys()


def test_random_search_tuner_no_params(dummy_pipeline_hyperparameters_small):
    tuner = RandomSearchTuner(dummy_pipeline_hyperparameters_small, random_state=random_state, with_replacement=False)
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            tuner.propose()


def test_random_search_tuner_with_replacement(dummy_pipeline_hyperparameters):
    tuner = RandomSearchTuner(dummy_pipeline_hyperparameters, random_state=random_state, with_replacement=True)
    for i in range(10):
        proposal = tuner.propose()
        assert isinstance(proposal, dict)
        assert proposal.keys() == dummy_pipeline_hyperparameters.keys()
        assert proposal['Mock Classifier'].keys() == dummy_pipeline_hyperparameters['Mock Classifier'].keys()


def test_random_search_tuner_basic(dummy_pipeline_hyperparameters,
                                   dummy_pipeline_hyperparameters_unicode):
    tuner = RandomSearchTuner(dummy_pipeline_hyperparameters, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'param a': 5,
            'param b': 8.442657485810175,
            'param c': 'option c',
            'param d': np.inf
        }
    }
    tuner.add(proposed_params, 0.5)

    tuner = RandomSearchTuner(dummy_pipeline_hyperparameters_unicode, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'param a': 5,
            'param b': 8.442657485810175,
            'param c': 'option c 💩',
            'param d': np.inf
        }
    }
    tuner.add(proposed_params, 0.5)


def test_random_search_tuner_space_types():
    tuner = RandomSearchTuner({'Mock Classifier': {'param a': (0, 10)}}, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'param a': 5}}

    tuner = RandomSearchTuner({'Mock Classifier': {'param a': (0, 10.0)}}, random_state=random_state)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'param a': 5.488135039273248}}


def test_random_search_tuner_invalid_space():
    value_error_text = 'Dimension has to be a list or tuple'
    bound_error_text = "has to be less than the upper bound"
    with pytest.raises(ValueError, match=value_error_text):
        RandomSearchTuner({'Mock Classifier': {'param a': False}}, random_state=random_state)
    with pytest.raises(ValueError, match=value_error_text):
        RandomSearchTuner({'Mock Classifier': {'param a': (0)}}, random_state=random_state)
    with pytest.raises(ValueError, match=bound_error_text):
        RandomSearchTuner({'Mock Classifier': {'param a': (1, 0)}}, random_state=random_state)
    with pytest.raises(ValueError, match=bound_error_text):
        RandomSearchTuner({'Mock Classifier': {'param a': (0, 0)}}, random_state=random_state)
