import pytest
import pandas as pd
import numpy as np
from evalml.problem_types import ProblemTypes, handle_problem_types, detect_problem_type


@pytest.fixture
def correct_problem_types():
    correct_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.MULTICLASS, ProblemTypes.BINARY]
    yield correct_problem_types


def test_handle_string(correct_problem_types):
    problem_types = ['regression', 'multiclass', 'binary']
    for problem_type in zip(problem_types, correct_problem_types):
        assert handle_problem_types(problem_type[0]) == problem_type[1]

    problem_type = 'fake'
    error_msg = 'Problem type \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_problem_types(problem_type) == ProblemTypes.REGRESSION


def test_handle_problem_types(correct_problem_types):
    for problem_type in correct_problem_types:
        assert handle_problem_types(problem_type) == problem_type


def test_handle_incorrect_type():
    error_msg = '`handle_problem_types` was not passed a str or ProblemTypes object'
    with pytest.raises(ValueError, match=error_msg):
        handle_problem_types(5)


def test_detect_problem_type_error():
    y_empty = pd.Series([])
    y_one_value = pd.Series([1, 1, 1, 1, 1, 1])
    y_nan = pd.Series([np.nan, np.nan, 1, 1, 1])

    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_empty)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_one_value)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_nan)


def test_detect_problem_type_binary():
    y_binary = pd.Series([1, 0, 1, 0, 0])
    y_bool = pd.Series([True, False, True, True, True])
    y_float = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    y_categorical = pd.Series(['yes', 'no', 'no', 'yes'])

    assert detect_problem_type(y_binary) == 'binary'
    assert detect_problem_type(y_bool) == 'binary'
    assert detect_problem_type(y_float) == 'binary'
    assert detect_problem_type(y_categorical) == 'binary'


def test_detect_problem_type_multiclass():
    y_multi = pd.Series([1, 2, 0, 2, 0, 0])
    y_categorical = pd.Series(['yes', 'no', 'maybe', 'no'])
    y_float = pd.Series([1, 2, 3.0, 2.0000, 1, 0, 0])

    assert detect_problem_type(y_multi) == 'multiclass'
    assert detect_problem_type(y_categorical) == 'multiclass'
    assert detect_problem_type(y_float) == 'multiclass'


def test_detect_problem_type_regression():
    y_regress = pd.Series([1.0, 2.1, 1.2, 0.3, 3.0, 2.3])
    y_mix = pd.Series([1, 0, 2, 3.000001])

    assert detect_problem_type(y_regress) == 'regression'
    assert detect_problem_type(y_mix) == 'regression'
