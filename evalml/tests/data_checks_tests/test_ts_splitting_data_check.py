import pandas as pd
import pytest

from evalml.data_checks import DataCheckMessageCode, TimeSeriesSplittingDataCheck


def test_time_series_splitting_data_check_raises_value_error():
    with pytest.raises(
        ValueError,
        match="Valid splitting of labels in time series",
    ):
        TimeSeriesSplittingDataCheck("time series regression", n_splits=3)


@pytest.mark.parametrize(
    "problem_type",
    ["time series binary", "time series multiclass"],
)
@pytest.mark.parametrize("is_valid", [True, False])
def test_time_series_param_data_check(is_valid, problem_type):
    X = None
    invalid_splits = {}

    if not is_valid:
        if problem_type == "time series binary":
            y = pd.Series([i % 2 if i < 25 else 1 for i in range(100)])
            invalid_splits = {
                1: {"Validation": [25, 50]},
                2: {"Validation": [50, 75]},
                3: {"Validation": [75, 100]},
            }
        elif problem_type == "time series multiclass":
            y = pd.Series([i % 3 if i > 65 else 2 for i in range(100)])
            invalid_splits = {
                1: {"Training": [0, 25], "Validation": [25, 50]},
                2: {"Training": [0, 50]},
            }
    else:
        if problem_type == "time series binary":
            y = pd.Series([i % 2 for i in range(100)])
        elif problem_type == "time series multiclass":
            y = pd.Series([i % 3 for i in range(100)])

    data_check = TimeSeriesSplittingDataCheck("time series binary", 3)
    messages = data_check.validate(X, y)
    code = DataCheckMessageCode.TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT.name

    if not is_valid:
        assert len(messages) == 1
        assert messages[0]["details"] == {
            "columns": None,
            "rows": None,
            "invalid_splits": invalid_splits,
        }
        assert messages[0]["code"] == code
    else:
        assert messages == []
