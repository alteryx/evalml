import pytest

from evalml import AutoRegressor
from evalml.tuners import GridSearchTuner, NoParamsException


def test_grid_search_tuner(X_y):
    X, y = X_y
    clf = AutoRegressor(objective="R2", max_pipelines=5, tuner=GridSearchTuner)
    clf.fit(X, y)


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
