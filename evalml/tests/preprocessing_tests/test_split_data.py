import pandas as pd
import pytest

from evalml.preprocessing import split_data, split_multiseries_data
from evalml.problem_types import (
    ProblemTypes,
    is_binary,
    is_multiclass,
    is_multiseries,
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
    multiseries_ts_data_unstacked,
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
        if is_multiseries(problem_type):
            X, y = multiseries_ts_data_unstacked

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
    if not is_multiseries(problem_type):
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
    else:
        assert isinstance(y_train, pd.DataFrame)
        assert isinstance(y_test, pd.DataFrame)
        pd.testing.assert_frame_equal(X_test, X[int(train_size) :], check_dtype=False)
        pd.testing.assert_frame_equal(y_test, y[int(train_size) :], check_dtype=False)

    if is_time_series(problem_type) and not is_multiseries(problem_type):
        pd.testing.assert_frame_equal(X_test, X[int(train_size) :], check_dtype=False)
        pd.testing.assert_series_equal(y_test, y[int(train_size) :], check_dtype=False)


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
@pytest.mark.parametrize("data_type", ["np", "pd", "ww"])
def test_split_data_defaults(
    problem_type,
    data_type,
    get_test_data_from_configuration,
    multiseries_ts_data_unstacked,
):
    X, y = get_test_data_from_configuration(
        data_type,
        problem_type,
        column_names=["numerical"],
        scale=10,
    )

    problem_configuration = None
    if is_time_series(problem_type):
        problem_configuration = {"gap": 1, "max_delay": 7, "time_index": "date"}
        if is_multiseries(problem_type):
            X, y = multiseries_ts_data_unstacked
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
        if not is_multiseries(problem_type):
            pd.testing.assert_series_equal(
                y_test,
                y[int(train_size) :],
                check_dtype=False,
            )
        else:
            pd.testing.assert_frame_equal(
                y_test,
                y[int(train_size) :],
                check_dtype=False,
            )


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


def test_split_data_calls_multiseries_error(multiseries_ts_data_stacked):
    X, y = multiseries_ts_data_stacked
    with pytest.raises(
        ValueError,
        match="requires problem_configuration for multiseries",
    ):
        split_data(X, y, problem_type="multiseries time series regression")

    with pytest.raises(
        ValueError,
        match="needs both series_id and time_index values in the problem_configuration",
    ):
        split_data(
            X,
            y,
            problem_type="multiseries time series regression",
            problem_configuration={"time_index": "date"},
        )


@pytest.mark.parametrize("no_features", [True, False])
@pytest.mark.parametrize("splitting_function", ["split_data", "split_multiseries_data"])
def test_split_multiseries_data(
    no_features,
    splitting_function,
    multiseries_ts_data_stacked,
):
    X, y = multiseries_ts_data_stacked

    if no_features:
        X = X[["date", "series_id"]]

    X_train_expected, X_holdout_expected = X[:-10], X[-10:]
    y_train_expected, y_holdout_expected = y[:-10], y[-10:]

    # Results should be identical whether split_multiseries_data is called through
    # split_data or directly
    if splitting_function == "split_data":
        X_train, X_holdout, y_train, y_holdout = split_data(
            X,
            y,
            problem_type="multiseries time series regression",
            problem_configuration={"time_index": "date", "series_id": "series_id"},
        )
    else:
        X_train, X_holdout, y_train, y_holdout = split_multiseries_data(
            X,
            y,
            "series_id",
            "date",
        )

    pd.testing.assert_frame_equal(
        X_train.sort_index(axis=1),
        X_train_expected.sort_index(axis=1),
    )
    pd.testing.assert_frame_equal(
        X_holdout.sort_index(axis=1),
        X_holdout_expected.sort_index(axis=1),
    )
    pd.testing.assert_series_equal(
        y_train,
        y_train_expected,
    )
    pd.testing.assert_series_equal(
        y_holdout,
        y_holdout_expected,
    )
