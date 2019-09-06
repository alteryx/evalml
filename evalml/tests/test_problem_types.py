import pytest

from evalml.problem_types import ProblemTypes, handle_problem_types


def test_handle_basic():
    assert handle_problem_types('multiclass') == ProblemTypes.MULTICLASS
    assert handle_problem_types('binary') == ProblemTypes.BINARY
    assert handle_problem_types('regression') == ProblemTypes.REGRESSION
    assert handle_problem_types(ProblemTypes.MULTICLASS) == ProblemTypes.MULTICLASS

    error_msg = 'Problem type \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_problem_types('fake') == ProblemTypes.regression


def test_handle_lists():
    pts = ['multiclass', 'binary']
    assert handle_problem_types(pts) == [ProblemTypes.MULTICLASS, ProblemTypes.BINARY]

    pts = ['regression']
    assert handle_problem_types(pts) == [ProblemTypes.REGRESSION]

    pts = ['fake', 'regression']
    error_msg = 'Problem type \'fake\' does not exist'
    with pytest.raises(KeyError, match=error_msg):
        handle_problem_types(pts) == ProblemTypes.regression
