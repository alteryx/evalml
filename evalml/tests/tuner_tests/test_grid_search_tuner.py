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


def test_grid_search_tuner_unique_values(dummy_component_hyperparameters, dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters)
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    generated_parameters = []
    for i in range(10):
        params = tuner.propose()
        generated_parameters.append(params)
    assert len(generated_parameters) == 10
    for i in range(10):
        assert generated_parameters[i].keys() == {'Mock Classifier'}
        assert generated_parameters[i]['Mock Classifier'].keys() == dummy_component_hyperparameters.keys()


def test_grid_search_tuner_no_params(dummy_component_hyperparameters_small, dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters_small)
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    error_text = "Grid search has exhausted all possible parameters."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            tuner.propose()


def test_grid_search_tuner_basic(dummy_component_hyperparameters,
                                 dummy_component_hyperparameters_unicode,
                                 dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters)
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'column a': 0,
            'column b': 0.0,
            'column c': 'option a',
            'column d': 'option a'
        }
    }
    tuner.add(proposed_params, 0.5)

    MockBinaryClassificationPipeline = dummy_binary_pipeline_class(dummy_component_hyperparameters_unicode)
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    proposed_params = tuner.propose()
    assert proposed_params == {
        'Mock Classifier': {
            'column a': 0,
            'column b': 0.0,
            'column c': 'option a 💩',
            'column d': 'option a'
        }
    }
    tuner.add(proposed_params, 0.5)


def test_grid_search_tuner_space_types(dummy_binary_pipeline_class):
    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({'column a': (0, 10)})
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'column a': 0}}

    MockBinaryClassificationPipeline = dummy_binary_pipeline_class({'column a': (0, 10.0)})
    tuner = GridSearchTuner(MockBinaryClassificationPipeline)
    proposed_params = tuner.propose()
    assert proposed_params == {'Mock Classifier': {'column a': 0}}


def test_grid_search_tuner_invalid_space(dummy_binary_pipeline_class):
    type_error_text = 'Invalid dimension type in tuner'
    bound_error_text = "Upper bound must be greater than lower bound. Parameter lower bound is 1 and upper bound is 0"
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner(dummy_binary_pipeline_class({'column a': False}))
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner(dummy_binary_pipeline_class({'column a': (0)}))
    with pytest.raises(ValueError, match=bound_error_text):
        GridSearchTuner(dummy_binary_pipeline_class({'column a': (1, 0)}))
