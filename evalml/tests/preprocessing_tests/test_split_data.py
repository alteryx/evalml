import warnings

import pytest
import woodwork as ww

from evalml.preprocessing import split_data
from evalml.problem_types import (
    ProblemTypes,
    is_binary,
    is_multiclass,
    is_regression,
    is_time_series
)


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_split_data(problem_type, data_type, X_y_binary, X_y_multi, X_y_regression, make_data_type):
    if is_binary(problem_type):
        X, y = X_y_binary
    if is_multiclass(problem_type):
        X, y = X_y_multi
    if is_regression(problem_type):
        X, y = X_y_regression
    problem_configuration = None
    if is_time_series(problem_type):
        problem_configuration = {'gap': 1, 'max_delay': 7}

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct, problem_type=problem_type,
                                                  problem_configuration=problem_configuration)
    test_size = len(X) * test_pct
    train_size = len(X) - test_size
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size
    assert isinstance(X_train, ww.DataTable)
    assert isinstance(X_test, ww.DataTable)
    assert isinstance(y_train, ww.DataColumn)
    assert isinstance(y_test, ww.DataColumn)


def test_split_data_raises_deprecated_random_state_warning(X_y_binary):
    X, y = X_y_binary
    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        split_data(X, y, test_size=0.2, problem_type="binary", random_state=0)
        assert str(warn[0].message).startswith("Argument 'random_state' has been deprecated in favor of 'random_seed'")

    with warnings.catch_warnings(record=True) as warn:
        warnings.simplefilter("always")
        split_data(X, y, test_size=0.2, problem_type="binary", random_seed=0)
        assert not warn
