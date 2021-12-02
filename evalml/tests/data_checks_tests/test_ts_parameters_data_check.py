import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckMessageCode,
    TimeSeriesParametersDataCheck,
)


def test_time_series_param_data_check_raises_value_error():
    with pytest.raises(
        ValueError,
        match="containing values for at least the date_index, gap, max_delay",
    ):
        TimeSeriesParametersDataCheck({}, n_splits=3)


@pytest.mark.parametrize(
    "gap,max_delay,forecast_horizon,n_obs,n_splits,is_valid",
    [
        [0, 5, 2, 100, 3, True],
        [0, 50, 1, 100, 3, False],
        [0, 24, 1, 100, 3, False],
        [0, 24, 1, 100, 2, True],
        [1, 23, 1, 100, 3, False],
        [1, 8, 2, 100, 9, False],
    ],
)
def test_time_series_param_data_check(
    gap, max_delay, forecast_horizon, n_obs, n_splits, is_valid
):

    config = {
        "gap": gap,
        "max_delay": max_delay,
        "forecast_horizon": forecast_horizon,
        "date_index": "date",
    }
    data_check = TimeSeriesParametersDataCheck(config, n_splits)
    X = pd.DataFrame({"feature": range(n_obs)})
    results = data_check.validate(X)
    code = DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT.name
    if not is_valid:
        assert len(results["errors"]) == 1
        assert results["errors"][0]["details"] == {
            "max_window_size": gap + max_delay + forecast_horizon,
            "min_split_size": n_obs // (n_splits + 1),
            "columns": None,
            "rows": None,
        }
        assert results["errors"][0]["code"] == code
    else:
        assert results == {"warnings": [], "errors": [], "actions": []}
