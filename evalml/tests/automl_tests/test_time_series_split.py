import pandas as pd
import pytest

from evalml.automl.data_splitters import TimeSeriesSplit


def test_time_series_split_init():
    ts_split = TimeSeriesSplit(gap=3, max_delay=4, n_folds=5)
    assert ts_split.get_n_splits() == 5

    with pytest.raises(ValueError, match="Both X and y cannot be None in TimeSeriesSplit.split"):
        _ = list(ts_split.split(X=None, y=None))

    with pytest.raises(ValueError, match="Both X and y cannot be None in TimeSeriesSplit.split"):
        _ = list(ts_split.split(X=pd.DataFrame(), y=pd.Series([])))


@pytest.mark.parametrize("max_delay,gap", [(0, 0), (1, 1), (2, 0), (4, 2)])
@pytest.mark.parametrize("X_none,y_none", [(False, False), (True, False), (False, True)])
def test_time_series_split(max_delay, gap, X_none, y_none):
    X = pd.DataFrame({"features": range(1, 32)})
    y = pd.Series(range(1, 32))

    # Splitter does not need a daterange index. We use a daterange index so that the
    # expected answer is easier to understand
    y.index = pd.date_range("2020-10-01", "2020-10-31")
    X.index = pd.date_range("2020-10-01", "2020-10-31")

    answer = [(pd.date_range("2020-10-01", f"2020-10-{10 + gap}"), pd.date_range(f"2020-10-{11 - max_delay}", f"2020-10-{17 + gap}")),
              (pd.date_range("2020-10-01", f"2020-10-{17 + gap}"), pd.date_range(f"2020-10-{18 - max_delay}", f"2020-10-{24 + gap}")),
              (pd.date_range("2020-10-01", f"2020-10-{24 + gap}"), pd.date_range(f"2020-10-{25 - max_delay}", "2020-10-31"))]

    if X_none:
        X = None
    if y_none:
        y = None

    ts_split = TimeSeriesSplit(gap=gap, max_delay=max_delay)
    for i, (train, test) in enumerate(ts_split.split(X, y)):
        if not X_none:
            X_train, X_test = X.iloc[train], X.iloc[test]
            pd.testing.assert_index_equal(X_train.index, answer[i][0])
            pd.testing.assert_index_equal(X_test.index, answer[i][1])
        if not y_none:
            y_train, y_test = y.iloc[train], y.iloc[test]
            pd.testing.assert_index_equal(y_train.index, answer[i][0])
            pd.testing.assert_index_equal(y_test.index, answer[i][1])
