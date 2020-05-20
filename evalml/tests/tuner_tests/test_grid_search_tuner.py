import pytest

from evalml import AutoRegressionSearch
from evalml.tuners import GridSearchTuner, NoParamsException
from evalml.tuners.tuner import Tuner


def test_grid_search_tuner_inheritance():
    assert issubclass(GridSearchTuner, Tuner)


def test_grid_search_tuner_automl(X_y):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=GridSearchTuner)
    clf.search(X, y)


def test_grid_search_tuner_automl_no_params(X_y, capsys):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=20, allowed_model_families=['linear_model'], tuner=GridSearchTuner)
    error_text = "Grid search has exhausted all possible parameters"
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


def test_grid_search_tuner_unique_values(dummy_pipeline_hyperparameters):
    tuner = GridSearchTuner(dummy_pipeline_hyperparameters)
    generated_parameters = []
    for i in range(10):
        params = tuner.propose()
        generated_parameters.append(params)
    assert len(generated_parameters) == 10
    for i in range(10):
        assert generated_parameters[i].keys() == dummy_pipeline_hyperparameters.keys()
        assert generated_parameters[i]['Mock Classifier'].keys() == dummy_pipeline_hyperparameters['Mock Classifier'].keys()


def test_grid_search_tuner_no_params(dummy_pipeline_hyperparameters_small):
    tuner = GridSearchTuner(dummy_pipeline_hyperparameters_small)
    error_text = "Grid search has exhausted all possible parameters."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            tuner.propose()


def test_grid_search_tuner_basic(dummy_pipeline_hyperparameters,
                                 dummy_pipeline_hyperparameters_unicode):
    tuner = GridSearchTuner(dummy_pipeline_hyperparameters)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'param a': 0,
            'param b': 0.0,
            'param c': 'option a',
            'param d': 'option a'
        }
    }
    tuner.add(proposed_params, 0.5)

    tuner = GridSearchTuner(dummy_pipeline_hyperparameters_unicode)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'param a': 0,
            'param b': 0.0,
            'param c': 'option a 💩',
            'param d': 'option a'
        }
    }
    tuner.add(proposed_params, 0.5)


def test_grid_search_tuner_space_types():
    tuner = GridSearchTuner({'Mock Classifier': {'param a': (0, 10)}})
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'param a': 0}}

    tuner = GridSearchTuner({'Mock Classifier': {'param a': (0, 10.0)}})
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'param a': 0}}


def test_grid_search_tuner_invalid_space():
    type_error_text = 'Invalid dimension type in tuner'
    bound_error_text = "Upper bound must be greater than lower bound. Parameter lower bound is 1 and upper bound is 0"
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner({'Mock Classifier': {'param a': False}})
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner({'Mock Classifier': {'param a': (0)}})
    with pytest.raises(ValueError, match=bound_error_text):
        GridSearchTuner({'Mock Classifier': {'param a': (1, 0)}})
