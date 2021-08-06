import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DateTimeFormatDataCheck,
)
from evalml.utils import infer_feature_types

datetime_format_check_name = DateTimeFormatDataCheck.name


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("uneven,type_errors", [(True, False), (False, True), (False, False)])
@pytest.mark.parametrize("datetime_loc", ["datetime_feature", "X_index", "y_index"])
def test_datetime_format_data_check_typeerror_uneven_intervals(
    uneven, input_type, type_errors, datetime_loc
):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))

    if type_errors:
        dates = range(30)
    else:
        dates = pd.date_range("2021-01-01", periods=30)

    if uneven:
        dates = list(dates)
        dates[-1] = "2021-01-31"
        dates = pd.DatetimeIndex(dates)

    datetime_column = "index"
    if datetime_loc == 'datetime_feature':
        X[datetime_loc] = dates
        datetime_column = datetime_loc
    elif datetime_loc == 'X_index':
        X.index = dates
    else:
        y.index = dates

    if input_type == "ww":
        X.ww.init()
        y.ww.init()

    datetime_format_check = DateTimeFormatDataCheck(datetime_column=datetime_column)

    if type_errors:
        if datetime_loc == "datetime_feature":
            with pytest.raises(
                    TypeError, match="That column does not contain datetime",
            ):
                datetime_format_check.validate(X, y)
        else:
            with pytest.raises(
                    TypeError, match="Either X or y has to have datetime information in its index.",
            ):
                datetime_format_check.validate(X, y)
    else:
        if not uneven:
            assert datetime_format_check.validate(X, y) == {
                "warnings": [],
                "errors": [],
                "actions": [],
            }
        else:
            message = datetime_loc if datetime_loc == "datetime_feature" else "either index"
            assert datetime_format_check.validate(X, y) == {
                "errors": [
                    DataCheckError(
                        message=f"No frequency could be detected in {message}, possibly due to uneven intervals.",
                        data_check_name=datetime_format_check_name,
                        message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                    ).to_dict()
                ],
                "warnings": [],
                "actions": [],
            }


@pytest.mark.parametrize("sort_order", ["increasing", "decreasing", "mixed"])
@pytest.mark.parametrize("datetime_loc", ["datetime_feature", "X_index", "y_index"])
def test_datetime_format_data_check_monotonic(datetime_loc, sort_order):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))
    dates = pd.date_range("2021-01-01", periods=30)

    if sort_order == "decreasing":
        dates = dates[::-1]
    elif sort_order == "mixed":
        dates = dates[:5].append(dates[10:]).append(dates[5:10])

    datetime_column = "index"
    if datetime_loc == 'datetime_feature':
        X[datetime_loc] = dates
        datetime_column = 'datetime_feature'
    elif datetime_loc == 'X_index':
        X.index = dates
    else:
        y.index = dates

    datetime_format_check = DateTimeFormatDataCheck(datetime_column=datetime_column)

    if sort_order != "mixed":
        assert datetime_format_check.validate(X, y) == {
            "warnings": [],
            "errors": [],
            "actions": [],
        }
    else:
        message = datetime_loc if datetime_loc == "datetime_feature" else "either index"
        assert datetime_format_check.validate(X, y) == {
            "errors": [
                DataCheckError(
                    message=f"No frequency could be detected in {message}, possibly due to uneven intervals.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                ).to_dict(),
                DataCheckError(
                    message="Datetime values must be monotonically increasing or decreasing.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                ).to_dict()
            ],
            "warnings": [],
            "actions": [],
        }
