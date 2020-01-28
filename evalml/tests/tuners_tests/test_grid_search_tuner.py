import pytest

from evalml import AutoRegressionSearch
from evalml.tuners import GridSearchTuner, NoParamsException


def test_autoreg_grid_search_tuner(X_y):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=GridSearchTuner)
    clf.search(X, y)


def test_autoreg_grid_search_tuner_no_params(X_y, capsys):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=20, model_types=['linear_model'], tuner=GridSearchTuner)
    error_text = "Grid search has exhausted all possible parameters"
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


def test_grid_search_tuner_unique_values(example_space):
    tuner = GridSearchTuner(example_space, random_state=0)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert len(generated_parameters) == 10


def test_grid_search_tuner_no_params(small_space):
    tuner = GridSearchTuner(small_space, random_state=0)
    generated_parameters = set()
    error_text = "Grid search has exhausted all possible parameters."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            params = tuner.propose()
            generated_parameters.add(tuple(params))
