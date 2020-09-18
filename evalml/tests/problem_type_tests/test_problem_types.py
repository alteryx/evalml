import numpy as np
import pandas as pd
import pytest

from evalml.problem_types import (
    ProblemTypes,
    detect_problem_type,
    handle_problem_types
)


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
    y_nan = pd.Series([np.nan, np.nan])

    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_empty)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_one_value)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_nan)


def test_detect_problem_type_binary():
    y_binary = pd.Series([1, 0, 1, 0, 0, 1])
    y_bool = pd.Series([True, False, True, True, True])
    y_float = pd.Series([1.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    y_categorical = pd.Series(['yes', 'no', 'no', 'yes'])
    y_null = pd.Series([None, 0, 1, 1, 1])

    assert detect_problem_type(y_binary) == 'binary'
    assert detect_problem_type(y_bool) == 'binary'
    assert detect_problem_type(y_float) == 'binary'
    assert detect_problem_type(y_categorical) == 'binary'
    assert detect_problem_type(y_null) == 'binary'


def test_detect_problem_type_multiclass():
    y_multi = pd.Series([1, 2, 0, 2, 0, 0, 1])
    y_categorical = pd.Series(['yes', 'no', 'maybe', 'no'], dtype='category')
    y_classes = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9] * 5)
    y_float = pd.Series([1.0, 2, 3])
    y_obj = pd.Series(["y", "n", "m"])

    assert detect_problem_type(y_multi) == 'multiclass'
    assert detect_problem_type(y_categorical) == 'multiclass'
    assert detect_problem_type(y_classes) == 'multiclass'
    assert detect_problem_type(y_float) == 'multiclass'
    assert detect_problem_type(y_obj) == 'multiclass'


def test_detect_problem_type_regression():
    y_values = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    y_float = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11.0])

    assert detect_problem_type(y_values) == 'regression'
    assert detect_problem_type(y_float) == 'regression'
    

def test_nan_none_na():
    y_none = pd.Series([None])
    y_pdna = pd.Series([pd.NA])
    y_nan = pd.Series([np.nan])
    y_all_null = pd.Series([None, pd.NA, np.nan])

    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_none)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_pdna)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_nan)
    with pytest.raises(ValueError, match="Less than 2"):
        detect_problem_type(y_all_null)
