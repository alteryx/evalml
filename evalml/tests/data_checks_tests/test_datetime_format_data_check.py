import numpy as np
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
    "issue", ["redundant", "missing", "uneven", "type_errors", None]
)
@pytest.mark.parametrize("datetime_loc", [1, "X_index", "y_index"])
def test_datetime_format_data_check_typeerror_uneven_intervals(
    issue, input_type, datetime_loc
):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))

    if issue == "type_errors":
        dates = range(30)
    else:
        dates = pd.date_range("2021-01-01", periods=30)

    if issue == "missing":
        # Skips 2021-01-30 and appends 2021-01-31, skipping a date and triggering the error
        dates = pd.date_range("2021-01-01", periods=29).append(
            pd.date_range("2021-01-31", periods=1)
        )
    if issue == "uneven":
        dates = pd.DatetimeIndex(
            pd.date_range("2021-01-01", periods=6).append(
                pd.date_range("2021-01-07", periods=24, freq="H")
            )
        )
    if issue == "redundant":
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

    if issue == "type_errors":
        assert datetime_format_check.validate(X, y) == [
            DataCheckError(
                message=f"Datetime information could not be found in the data, or was not in a supported datetime format.",
                data_check_name=datetime_format_check_name,
                message_code=DataCheckMessageCode.DATETIME_INFORMATION_NOT_FOUND,
            ).to_dict()
        ]
    else:
        col_name = datetime_loc if datetime_loc == 1 else "either index"
        if issue is None:
            assert datetime_format_check.validate(X, y) == []
        elif issue == "missing":
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"Column '{col_name}' has datetime values missing between start and end date.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict()
            ]
        elif issue == "redundant":
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"Column '{col_name}' has more than one row with the same datetime value.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                ).to_dict()
            ]
        else:
            col_name = datetime_loc if datetime_loc == 1 else "either index"
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"Column '{col_name}' has datetime values missing between start and end date.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict(),
                DataCheckError(
                    message=f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                ).to_dict(),
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
            message=f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals.",
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
    elif n_missing == 5:
        # A chunk of 5 missing days in a row
        dates = dates.append(pd.date_range("2021-01-21", periods=15))
    else:
        # Some chunks missing and some alone missing
        dates = dates.append(pd.date_range("2021-01-20", periods=18)).drop("2021-01-27")
        dates = dates.drop("2021-01-22")
        dates = dates.drop("2021-01-05")

    X["dates"] = dates
    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict()
    ]


def test_datetime_format_data_check_multiple_errors():
    dates = (
        pd.date_range("2021-01-01", periods=9).tolist()
        + ["2021-01-31", "2021-02-02", "2021-02-04"]
        + pd.date_range("2021-02-05", periods=9).tolist()
    )
    X = pd.DataFrame({"dates": dates})
    y = pd.Series(range(21))
    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        DataCheckError(
            message=f"No frequency could be detected in column 'dates', possibly due to uneven intervals.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
        ).to_dict(),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=9).tolist()
        + ["2021-01-09", "2021-01-31", "2021-02-02", "2021-02-04"]
        + pd.date_range("2021-02-05", periods=9).tolist()
    )
    X = pd.DataFrame({"dates": dates})

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        DataCheckError(
            message=f"No frequency could be detected in column 'dates', possibly due to uneven intervals.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
        ).to_dict(),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=15)
        .drop("2021-01-05")
        .append(pd.date_range("2021-01-15", periods=16))
    )
    X = pd.DataFrame({"dates": dates})

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
    ]


def test_datetime_format_unusual_interval():
    dates = pd.date_range(start="2021-01-01", periods=20, freq="4D")
    X = pd.DataFrame({"dates": dates})
    y = pd.Series(range(20))

    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")
    assert datetime_format_check.validate(X, y) == []

    expected = [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict()
    ]
    dates = dates.drop("2021-01-09")
    X = pd.DataFrame({"dates": dates})
    assert datetime_format_check.validate(X, y) == expected

    expected = [
        DataCheckError(
            message=f"Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
    ]
    dates = dates.append(pd.date_range("2021-03-18", periods=2, freq="4D"))
    X = pd.DataFrame({"dates": dates})
    assert datetime_format_check.validate(X, y) == expected


def test_datetime_format_nan_data_check_error():
    dates = pd.Series(pd.date_range(start="2021-01-01", periods=20))
    dates[0] = np.NaN
    X = pd.DataFrame(dates, columns=["date"])

    expected = [
        DataCheckError(
            message="Input datetime column 'date' contains NaN values. Please impute NaN values or drop these rows.",
            data_check_name=DateTimeFormatDataCheck.name,
            message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
        ).to_dict()
    ]

    dt_nan_check = DateTimeFormatDataCheck(datetime_column="date")
    assert dt_nan_check.validate(X, pd.Series()) == expected

    dates[5] = pd.to_datetime("2021-01-05")
    X = pd.DataFrame(dates, columns=["date"])

    expected.extend(
        [
            DataCheckError(
                message=f"Column 'date' has more than one row with the same datetime value.",
                data_check_name=datetime_format_check_name,
                message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
            ).to_dict(),
            DataCheckError(
                message=f"Column 'date' has datetime values missing between start and end date.",
                data_check_name=datetime_format_check_name,
                message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
            ).to_dict(),
        ]
    )

    assert dt_nan_check.validate(X, pd.Series()) == expected


def test_datetime_nan_check_ww():
    dt_nan_check = DateTimeFormatDataCheck(datetime_column="dates")
    y = pd.Series()

    expected = [
        DataCheckError(
            message="Input datetime column 'dates' contains NaN values. Please impute NaN values or drop these rows.",
            data_check_name=DateTimeFormatDataCheck.name,
            message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
        ).to_dict()
    ]

    dates = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-08"))
    dates[0] = np.datetime64("NaT")

    ww_input = pd.DataFrame(dates, columns=["dates"])
    ww_input.ww.init()
    assert dt_nan_check.validate(ww_input, y) == expected
