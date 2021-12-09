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
        TimeSeriesParametersDataCheck({}, "time series regression", n_splits=3)


@pytest.mark.parametrize(
    "gap,max_delay,forecast_horizon,n_obs,n_splits,problem_type,y_data",
    [
        [0, 5, 2, 100, 3, "time series binary", None],
        [0, 25, 2, 100, 3, "time series multiclass", None],
        [0, 50, 1, 100, 3, "time series binary", "valid"],
        [0, 24, 1, 100, 3, "time series multiclass", "invalid"],
        [0, 24, 1, 100, 2, "time series multiclass", "valid"],
        [0, 24, 1, 100, 2, "time series multiclass", "invalid"],
        [1, 23, 1, 100, 3, "time series binary", "invalid"],
        [1, 8, 2, 100, 9, "time series multiclass", "valid"],
    ],
)
def test_time_series_param_data_check(
    gap, max_delay, forecast_horizon, n_obs, n_splits, problem_type, y_data
):
    y = None
    if y_data == "valid":
        if problem_type == "time series binary":
            y = pd.Series([1, 0, 1, 1, 0] * 20)
        elif problem_type == "time series multiclass":
            y = pd.Series([1, 0, 1, 2, 0] * 20)
    elif y_data == "invalid":
        if problem_type == "time series binary":
            y = pd.Series([1 if i < 50 else 2 for i in range(100)])
        elif problem_type == "time series multiclass":
            y = pd.Series([1 if i < 50 else 2 if i < 70 else 3 for i in range(100)])

    split_size = n_obs // (n_splits + 1)
    window_size = gap + max_delay + forecast_horizon

    config = {
        "gap": gap,
        "max_delay": max_delay,
        "forecast_horizon": forecast_horizon,
        "date_index": "date",
    }
    data_check = TimeSeriesParametersDataCheck(config, problem_type, n_splits)
    X = pd.DataFrame({"feature": range(n_obs)})
    results = data_check.validate(X, y)
    code_params = (
        DataCheckMessageCode.TIMESERIES_PARAMETERS_NOT_COMPATIBLE_WITH_SPLIT.name
    )
    code_target = DataCheckMessageCode.TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT.name

    if y_data == "invalid" and split_size <= window_size:
        assert len(results["errors"]) == 2
    elif y_data == "invalid":
        assert len(results["errors"]) == 1
        assert results["errors"][0]["details"] == {
            "invalid_splits": [i + 1 for i in range(n_splits)],
            "columns": None,
            "rows": None,
        }
        assert results["errors"][0]["code"] == code_target
        assert "Time Series Binary" in results["errors"][0]["message"]
        assert "collect more data. " not in results["errors"][0]["message"]
    elif split_size <= window_size:
        assert len(results["errors"]) == 1
        assert results["errors"][0]["details"] == {
            "max_window_size": window_size,
            "min_split_size": split_size,
            "columns": None,
            "rows": None,
        }
        assert results["errors"][0]["code"] == code_params
        assert (
            "Please use a smaller number of splits" in results["errors"][0]["message"]
        )
        assert "Time Series Binary" not in results["errors"][0]["message"]
    else:
        assert results == {"warnings": [], "errors": [], "actions": []}
