import pytest

from evalml.problem_types import ProblemTypes, handle_problem_types


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
    for problem_type in zip(correct_problem_types, correct_problem_types):
        assert handle_problem_types(problem_type[0]) == problem_type[1]


def test_handle_incorrect_type():
    error_msg = '`handle_problem_types` was not passed a str or ProblemTypes object'
    with pytest.raises(ValueError, match=error_msg):
        handle_problem_types(5)
