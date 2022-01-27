import pandas as pd
import pytest

from evalml.preprocessing.data_splitters import TimeSeriesSplit


def test_time_series_split_init():
    ts_split = TimeSeriesSplit(gap=3, max_delay=4, n_splits=5, time_index=None)
    assert ts_split.get_n_splits() == 5

    with pytest.raises(
        ValueError,
        match="Both X and y cannot be None or empty in TimeSeriesSplit.split",
    ):
        _ = list(ts_split.split(X=None, y=None))

    with pytest.raises(
        ValueError,
        match="Both X and y cannot be None or empty in TimeSeriesSplit.split",
    ):
        _ = list(ts_split.split(X=pd.DataFrame(), y=pd.Series([])))


@pytest.mark.parametrize(
    "gap,max_delay,forecast_horizon,n_splits",
    [[7, 3, 1, 5], [0, 8, 2, 3], [5, 4, 2, 4]],
)
def test_time_series_split_n_splits_too_big(gap, max_delay, forecast_horizon, n_splits):
    splitter = TimeSeriesSplit(
        gap=gap,
        max_delay=max_delay,
        forecast_horizon=forecast_horizon,
        n_splits=n_splits,
        time_index="date",
    )
    X = pd.DataFrame({"features": range(15)})
    with pytest.raises(ValueError, match="Please use a smaller number of splits"):
        list(splitter.split(X))


@pytest.mark.parametrize(
    "max_delay,gap,forecast_horizon,time_index",
    [
        (0, 0, 1, "Date"),
        (1, 0, 1, None),
        (2, 0, 1, "Date"),
        (0, 3, 1, None),
        (1, 1, 1, "Date"),
    ],
)
@pytest.mark.parametrize("y_none", [False, True])
def test_time_series_split(max_delay, gap, forecast_horizon, time_index, y_none):
    X = pd.DataFrame({"features": range(1, 32)})
    y = pd.Series(range(1, 32))

    # Splitter does not need a daterange index. We use a daterange index so that the
    # expected answer is easier to understand
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    if time_index:
        X[time_index] = pd.date_range("2020-10-01", "2020-10-31")
    else:
        X.index = pd.date_range("2020-10-01", "2020-10-31")

    answer = [
        (
            pd.date_range("2020-10-01", f"2020-10-28"),
            pd.date_range(f"2020-10-29", f"2020-10-29"),
        ),
        (
            pd.date_range("2020-10-01", f"2020-10-29"),
            pd.date_range(f"2020-10-30", f"2020-10-30"),
        ),
        (
            pd.date_range("2020-10-01", f"2020-10-30"),
            pd.date_range(f"2020-10-31", "2020-10-31"),
        ),
    ]
    answer_dt = [
        (pd.Index(range(28)), pd.Index(range(28, 29))),
        (pd.Index(range(29)), pd.Index(range(29, 30))),
        (pd.Index(range(30)), pd.Index(range(30, 31))),
    ]

    if y_none:
        y = None

    ts_split = TimeSeriesSplit(
        gap=gap,
        max_delay=max_delay,
        forecast_horizon=forecast_horizon,
        time_index=time_index,
    )
    for i, (train, test) in enumerate(ts_split.split(X, y)):
        X_train, X_test = X.iloc[train], X.iloc[test]
        if time_index:
            pd.testing.assert_index_equal(X_train.index, answer_dt[i][0])
            pd.testing.assert_index_equal(X_test.index, answer_dt[i][1])
        else:
            pd.testing.assert_index_equal(X_train.index, answer[i][0])
            pd.testing.assert_index_equal(X_test.index, answer[i][1])
        if not y_none:
            y_train, y_test = y.iloc[train], y.iloc[test]
            pd.testing.assert_index_equal(y_train.index, answer[i][0])
            pd.testing.assert_index_equal(y_test.index, answer[i][1])
