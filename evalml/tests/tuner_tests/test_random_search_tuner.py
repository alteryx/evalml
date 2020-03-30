import pytest

from evalml import AutoRegressionSearch
from evalml.tests.tuner_tests.tuner_test_utils import (
    assert_params_almost_equal
)
from evalml.tuners import NoParamsException, RandomSearchTuner
from evalml.tuners.tuner import Tuner


def test_autoreg_random_search_tuner(X_y):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=RandomSearchTuner)
    clf.search(X, y)


def test_autoreg_random_search_tuner_no_params(X_y, capsys):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=20, allowed_model_families=['linear_model'], tuner=RandomSearchTuner)
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


random_state = 0


def test_random_search_tuner_unique_values(test_space):
    tuner = RandomSearchTuner(test_space, random_state=0, with_replacement=True)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert len(generated_parameters) == 10
    assert len(list(generated_parameters)[0]) == 4


def test_random_search_tuner_no_params(test_space_small):
    tuner = RandomSearchTuner(test_space_small, random_state=0, with_replacement=False)
    generated_parameters = set()
    error_text = "Cannot create a unique set of unexplored parameters. Try expanding the search space."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            params = tuner.propose()
            generated_parameters.add(tuple(params))


def test_random_search_tuner_basic(test_space, test_space_unicode):
    tuner = RandomSearchTuner(test_space, random_state=random_state)
    assert isinstance(tuner, Tuner)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, 'option_c'])
    tuner.add(proposed_params, 0.5)

    tuner = RandomSearchTuner(test_space_unicode, random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, 'option_c 💩'])
    tuner.add(proposed_params, 0.5)


def test_random_search_tuner_space_types():
    tuner = RandomSearchTuner([(0, 10)], random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])

    tuner = RandomSearchTuner([(0, 10.0)], random_state=random_state)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])


def test_random_search_tuner_invalid_space():
    with pytest.raises(TypeError):
        RandomSearchTuner(False)
    with pytest.raises(ValueError):
        RandomSearchTuner([(0)])
    with pytest.raises(ValueError):
        RandomSearchTuner(((0, 1)))
    with pytest.raises(ValueError):
        RandomSearchTuner([(0, 0)])
