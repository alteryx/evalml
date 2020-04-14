import pytest

from evalml import AutoRegressionSearch
from evalml.tests.tuner_tests.tuner_test_utils import (
    assert_params_almost_equal
)
from evalml.tuners import GridSearchTuner, NoParamsException
from evalml.tuners.tuner import Tuner


def test_autoreg_grid_search_tuner(X_y):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=5, tuner=GridSearchTuner)
    clf.search(X, y)


def test_autoreg_grid_search_tuner_no_params(X_y, capsys):
    X, y = X_y
    clf = AutoRegressionSearch(objective="R2", max_pipelines=20, allowed_model_families=['linear_model'], tuner=GridSearchTuner)
    error_text = "Grid search has exhausted all possible parameters"
    with pytest.raises(NoParamsException, match=error_text):
        clf.search(X, y)


def test_grid_search_tuner_unique_values(test_space):
    tuner = GridSearchTuner(test_space)
    generated_parameters = set()
    for i in range(10):
        params = tuner.propose()
        generated_parameters.add(tuple(params))
    assert len(generated_parameters) == 10
    assert len(list(generated_parameters)[0]) == 4


def test_grid_search_tuner_no_params(test_space_small):
    tuner = GridSearchTuner(test_space_small)
    generated_parameters = set()
    error_text = "Grid search has exhausted all possible parameters."
    with pytest.raises(NoParamsException, match=error_text):
        for i in range(10):
            params = tuner.propose()
            generated_parameters.add(tuple(params))


def test_grid_search_tuner_basic(test_space, test_space_unicode):
    tuner = GridSearchTuner(test_space)
    assert isinstance(tuner, Tuner)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, 'option_c'])
    tuner.add(proposed_params, 0.5)

    tuner = GridSearchTuner(test_space_unicode)
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5, 8.442657485810175, u'option_c 💩'])
    tuner.add(proposed_params, 0.5)


def test_grid_search_tuner_space_types():
    tuner = GridSearchTuner([(0, 10)])
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])

    tuner = GridSearchTuner([(0, 10.0)])
    proposed_params = tuner.propose()
    assert_params_almost_equal(proposed_params, [5.928446182250184])


def test_grid_search_tuner_invalid_space():
    iterable_error = '\'bool\' object is not iterable'
    type_error_text = 'Invalid dimension type in tuner'
    bound_error_text = "Upper bound must be greater than lower bound. Parameter lower bound is 1 and upper bound is 0"
    with pytest.raises(TypeError, match=iterable_error):
        GridSearchTuner(False)
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner([(0)])
    with pytest.raises(TypeError, match=type_error_text):
        GridSearchTuner(((0, 1)))
    with pytest.raises(ValueError, match=bound_error_text):
        GridSearchTuner([(1, 0)])
