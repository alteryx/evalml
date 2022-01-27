import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DateTimeFormatDataCheck,
)

datetime_format_check_name = DateTimeFormatDataCheck.name


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize(
    "redundant,missing,uneven,type_errors",
    [
        (False, False, True, False),
        (False, False, False, True),
        (False, False, False, False),
        (False, True, False, False),
        (True, False, False, False),
    ],
)
@pytest.mark.parametrize("datetime_loc", [1, "X_index", "y_index"])
def test_datetime_format_data_check_typeerror_uneven_intervals(
    redundant, missing, uneven, input_type, type_errors, datetime_loc
):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))

    if type_errors:
        dates = range(30)
    else:
        dates = pd.date_range("2021-01-01", periods=30)

    if missing:
        # Skips 2021-01-30 and appends 2021-01-31, skipping a date and triggering the error
        dates = pd.date_range("2021-01-01", periods=29).append(
            pd.date_range("2021-01-31", periods=1)
        )
    if uneven:
        dates = pd.DatetimeIndex(
            pd.date_range("2021-01-01", periods=6).append(
                pd.date_range("2021-01-07", periods=24, freq="H")
            )
        )
    if redundant:
        dates = pd.date_range("2021-01-01", periods=29).append(
            pd.date_range("2021-01-29", periods=1)
        )

    datetime_column = "index"
    if datetime_loc == 1:
        X[datetime_loc] = dates
        datetime_column = datetime_loc
    elif datetime_loc == "X_index":
        X.index = dates
    else:
        y.index = dates

    if input_type == "ww":
        X.ww.init()
        y.ww.init()

    datetime_format_check = DateTimeFormatDataCheck(datetime_column=datetime_column)

    if type_errors:
        assert datetime_format_check.validate(X, y) == [
            DataCheckError(
                message=f"Datetime information could not be found in the data, or was not in a supported datetime format.",
                data_check_name=datetime_format_check_name,
                message_code=DataCheckMessageCode.DATETIME_INFORMATION_NOT_FOUND,
            ).to_dict()
        ]
    else:
        col_name = datetime_loc if datetime_loc == 1 else "either index"
        if not (uneven or missing or redundant):
            assert datetime_format_check.validate(X, y) == []
        elif missing:
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"{col_name} has datetime values missing between start and end date around row(s) [28]",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict()
            ]
        elif redundant:
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"{col_name} has more than one row with the same datetime value",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                ).to_dict()
            ]
        else:
            col_name = datetime_loc if datetime_loc == 1 else "either index"
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"No frequency could be detected in {col_name}, possibly due to uneven intervals.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                ).to_dict()
            ]


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
    if datetime_loc == "datetime_feature":
        X[datetime_loc] = dates
        datetime_column = "datetime_feature"
    elif datetime_loc == "X_index":
        X.index = dates
    else:
        y.index = dates

    datetime_format_check = DateTimeFormatDataCheck(datetime_column=datetime_column)

    if sort_order == "increasing":
        assert datetime_format_check.validate(X, y) == []
    else:
        col_name = (
            datetime_loc if datetime_loc == "datetime_feature" else "either index"
        )
        freq_error = DataCheckError(
            message=f"No frequency could be detected in {col_name}, possibly due to uneven intervals.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
        ).to_dict()
        mono_error = DataCheckError(
            message="Datetime values must be sorted in ascending order.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
        ).to_dict()
        if sort_order == "decreasing":
            assert datetime_format_check.validate(X, y) == [mono_error]
        else:
            assert datetime_format_check.validate(X, y) == [freq_error, mono_error]


@pytest.mark.parametrize("n_missing", [2, 5, 7])
def test_datetime_format_data_check_multiple_missing(n_missing):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))

    dates = pd.date_range("2021-01-01", periods=15)
    if n_missing == 2:
        # Two missing dates in separate spots
        dates = dates.append(pd.date_range("2021-01-17", periods=16)).drop("2021-01-22")
        missing = [14, 19]
    elif n_missing == 5:
        # A chunk of 5 missing days in a row
        dates = dates.append(pd.date_range("2021-01-21", periods=15))
        missing = [14]
    else:
        # Some chunks missing and some alone missing
        missing = [3, 13, 15, 19]
        dates = dates.append(pd.date_range("2021-01-20", periods=18)).drop("2021-01-27")
        dates = dates.drop("2021-01-22")
        dates = dates.drop("2021-01-05")

    X["dates"] = dates
    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"dates has datetime values missing between start and end date around row(s) {missing}",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict()
    ]
