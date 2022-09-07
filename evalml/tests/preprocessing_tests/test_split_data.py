import pandas as pd
import pytest

from evalml.preprocessing import split_data
from evalml.problem_types import (
    ProblemTypes,
    is_binary,
    is_multiclass,
    is_regression,
    is_time_series,
)


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_split_data(
    problem_type,
    data_type,
    X_y_binary,
    X_y_multi,
    X_y_regression,
    make_data_type,
):
    if is_binary(problem_type):
        X, y = X_y_binary
    if is_multiclass(problem_type):
        X, y = X_y_multi
    if is_regression(problem_type):
        X, y = X_y_regression
    problem_configuration = None
    if is_time_series(problem_type):
        problem_configuration = {"gap": 1, "max_delay": 7, "time_index": "date"}

    X = make_data_type(data_type, X)
    y = make_data_type(data_type, y)

    test_pct = 0.25
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_pct,
        problem_type=problem_type,
        problem_configuration=problem_configuration,
    )
    test_size = len(X) * test_pct
    train_size = len(X) - test_size
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    if is_time_series(problem_type):
        pd.testing.assert_frame_equal(X_test, X[int(train_size) :], check_dtype=False)
        pd.testing.assert_series_equal(y_test, y[int(train_size) :], check_dtype=False)


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_split_data_defaults(problem_type, data_type, get_test_data_from_configuration):
    X, y = get_test_data_from_configuration(
        data_type,
        problem_type,
        column_names=["numerical"],
        scale=10,
    )

    problem_configuration = None
    if is_time_series(problem_type):
        problem_configuration = {"gap": 1, "max_delay": 7, "time_index": "date"}
        test_pct = 0.1
    else:
        test_pct = 0.2
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        problem_type=problem_type,
        problem_configuration=problem_configuration,
    )
    test_size = len(X) * test_pct
    train_size = len(X) - test_size
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size

    if is_time_series(problem_type):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            y = pd.Series(y)
        pd.testing.assert_frame_equal(X_test, X[int(train_size) :], check_dtype=False)
        pd.testing.assert_series_equal(y_test, y[int(train_size) :], check_dtype=False)


@pytest.mark.parametrize("test", ["fh_limitation", "no_fh_limitation"])
def test_split_data_ts(test, X_y_regression):
    X, y = X_y_regression

    if test == "no_fh_limitation":
        test_pct = 0.1
        fh = 5
        test_size = len(X) * test_pct
        train_size = len(X) - test_size
    elif test == "fh_limitation":
        fh = 25
        test_size = fh
        train_size = len(X) - fh

    problem_configuration = {
        "gap": 1,
        "max_delay": 7,
        "forecast_horizon": fh,
        "time_index": "date",
    }
    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        problem_type="time series regression",
        problem_configuration=problem_configuration,
    )
    assert len(X_train) == train_size
    assert len(X_test) == test_size
    assert len(y_train) == train_size
    assert len(y_test) == test_size
