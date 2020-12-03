import pandas as pd
import pytest
import woodwork as ww

from evalml.preprocessing import split_data
from evalml.problem_types import ProblemTypes


@pytest.mark.parametrize("problem_type", [ProblemTypes.BINARY, ProblemTypes.MULTICLASS, ProblemTypes.REGRESSION])
@pytest.mark.parametrize("data_type", ['np', 'pd', 'ww'])
def test_split_regression(problem_type, data_type, X_y_binary, X_y_multi, X_y_regression):
    if problem_type == ProblemTypes.BINARY:
        X, y = X_y_multi
    if problem_type == ProblemTypes.MULTICLASS:
        X, y = X_y_binary
    if problem_type == ProblemTypes.REGRESSION:
        X, y = X_y_regression

    if data_type != "np":
        X = pd.DataFrame(X)
        y = pd.Series(y)
    if data_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)

    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_pct, regression=True)
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
