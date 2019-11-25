import pytest
from skopt.space import Integer, Real

from evalml.tuners import GridSearchTuner, RandomSearchTuner


@pytest.fixture
def example_space():
    list_of_space = list()
    list_of_space.append(Integer(5, 100))
    list_of_space.append(Real(0.01, 10))
    list_of_space.append(['most_frequent', 'median', 'mean'])
    return list_of_space


def test_random_search_tuner(example_space):
    tuner = RandomSearchTuner(example_space, random_state=0, check_duplicates=True)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert list(generated_parameters) == 10


def test_grid_search_tuner(example_space):
    tuner = GridSearchTuner(example_space, random_state=0, check_duplicates=True)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert list(generated_parameters) == 10
