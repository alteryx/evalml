import pytest

from evalml.problem_types import ProblemTypes, handle_problem_types


@pytest.fixture
def correct_pts():
    correct_pts = [ProblemTypes.REGRESSION, ProblemTypes.MULTICLASS, ProblemTypes.BINARY]
    yield correct_pts


def test_handle_string(correct_pts):
    pts = ['regression', 'multiclass', 'binary']
    for pt in zip(pts, correct_pts):
        assert handle_problem_types(pt[0]) == pt[1]

    pts = 'fake'
    error_msg = 'Problem type \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_problem_types(pts) == ProblemTypes.REGRESSION


def test_handle_problemtypes(correct_pts):
    for pt in zip(correct_pts, correct_pts):
        assert handle_problem_types(pt[0]) == pt[1]


def test_handle_incorrect_type():
    error_msg = '`handle_problem_types` was not passed a str or ProblemTypes object'
    with pytest.raises(ValueError, match=error_msg):
        handle_problem_types(5)s
