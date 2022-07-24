import pandas as pd
import pytest

from evalml.data_checks import DataCheckMessageCode, TimeSeriesParametersDataCheck


@pytest.mark.parametrize(
    "gap,max_delay,forecast_horizon,time_index",
    [[1, 1, 1, None], [None, None, None, None], ["missing", 1, 1, "dates"]],
)
def test_time_series_param_data_check_raises_value_error(
    gap,
    max_delay,
    forecast_horizon,
    time_index,
):
    if all(i is None for i in [gap, max_delay, forecast_horizon, time_index]):
        params = None
    elif any(i == "missing" for i in [gap, max_delay, forecast_horizon, time_index]):
        params = {
            "max_delay": max_delay,
            "forecast_horizon": forecast_horizon,
            "time_index": time_index,
        }
    else:
        params = {
            "gap": gap,
            "max_delay": max_delay,
            "forecast_horizon": forecast_horizon,
            "time_index": time_index,
        }
    with pytest.raises(
        ValueError,
        match="containing values for at least the time_index, gap, max_delay",
    ):
        TimeSeriesParametersDataCheck(params, n_splits=3)


@pytest.mark.parametrize(
    "gap,max_delay,forecast_horizon,n_obs,n_splits,is_valid",
    [
        [0, 5, 2, 100, 3, True],
        [0, 50, 15, 100, 3, False],
        [0, 24, 20, 100, 3, False],
        [0, 24, 1, 100, 2, True],
        [1, 23, 20, 100, 3, False],
        [1, 50, 5, 100, 9, False],
    ],
)
def test_time_series_param_data_check(
    gap,
    max_delay,
    forecast_horizon,
    n_obs,
    n_splits,
    is_valid,
):

    config = {
        "gap": gap,
        "max_delay": max_delay,
        "forecast_horizon": forecast_horizon,
        "time_index": "date",
    }
    data_check = TimeSeriesParametersDataCheck(config, n_splits)
    X = pd.DataFrame({"feature": range(n_obs)})
    messages = data_check.validate(X)
    code = DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT.name
    if not is_valid:
        assert len(messages) == 1
        assert messages[0]["details"] == {
            "max_window_size": gap + max_delay + forecast_horizon,
            "min_split_size": n_obs - (forecast_horizon * n_splits),
            "n_obs": n_obs,
            "n_splits": n_splits,
            "columns": None,
            "rows": None,
        }
        assert messages[0]["code"] == code
    else:
        assert messages == []
