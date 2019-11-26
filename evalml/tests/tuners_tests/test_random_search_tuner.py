import pytest

from evalml import AutoRegressor
from evalml.tuners import NoParamsException, RandomSearchTuner


def test_random_search_tuner(X_y):
    X, y = X_y
    clf = AutoRegressor(objective="R2", max_pipelines=5, tuner=RandomSearchTuner)
    clf.fit(X, y)


def test_random_search_tuner_unique_values(example_space):
    tuner = RandomSearchTuner(example_space, random_state=0, check_duplicates=True)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert len(generated_parameters) == 10


def test_random_search_tuner_no_params(small_space):
    tuner = RandomSearchTuner(small_space, random_state=0, check_duplicates=True)
    generated_parameters = set()
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            params = tuner.propose()
            generated_parameters.add(tuple(params))
