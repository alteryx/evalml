import numpy as np
import pandas as pd
import pytest
from woodwork.statistics_utils import infer_frequency

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DateTimeFormatDataCheck,
    DCAOParameterType,
)

datetime_format_check_name = DateTimeFormatDataCheck.name

WINDOW_LENGTH = 4
THRESHOLD = 0.4


def get_uneven_error(col_name, ww_payload):
    error = DataCheckError(
        message=f"A frequency was detected in column '{col_name}', but there are faulty datetime values that need to be addressed.",
        data_check_name=datetime_format_check_name,
        message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
        action_options=[
            DataCheckActionOption(
                DataCheckActionCode.REGULARIZE_AND_IMPUTE_DATASET,
                data_check_name=datetime_format_check_name,
                parameters={
                    "time_index": {
                        "parameter_type": DCAOParameterType.GLOBAL,
                        "type": "str",
                        "default_value": col_name,
                    },
                    "frequency_payload": {
                        "default_value": ww_payload,
                        "parameter_type": "global",
                        "type": "tuple",
                    },
                },
                metadata={"is_target": True},
            ),
        ],
    ).to_dict()
    return error


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize(
    "issue",
    ["redundant", "missing", "uneven", "type_errors", None],
)
@pytest.mark.parametrize("datetime_loc", [1, "X_index", "y_index"])
def test_datetime_format_data_check_typeerror_uneven_intervals(
    issue,
    input_type,
    datetime_loc,
):
    X, y = pd.DataFrame({"features": range(30)}), pd.Series(range(30))

    if issue == "type_errors":
        dates = range(30)
    else:
        dates = pd.date_range("2021-01-01", periods=30)

    if issue == "missing":
        # Skips 2021-01-25 and starts again at 2021-01-27, skipping a date and triggering the error
        dates = pd.date_range("2021-01-01", periods=25).append(
            pd.date_range("2021-01-27", periods=5),
        )
    if issue == "uneven":
        dates_1 = pd.date_range("2015-01-01", periods=5, freq="D")
        dates_2 = pd.date_range("2015-01-08", periods=3, freq="D")
        dates_3 = pd.DatetimeIndex(["2015-01-12"])
        dates_4 = pd.date_range("2015-01-15", periods=5, freq="D")
        dates_5 = pd.date_range("2015-01-22", periods=5, freq="D")
        dates_6 = pd.date_range("2015-01-29", periods=11, freq="M")

        dates = (
            dates_1.append(dates_2)
            .append(dates_3)
            .append(dates_4)
            .append(dates_5)
            .append(dates_6)
        )
    if issue == "redundant":
        dates = pd.date_range("2021-01-01", periods=29).append(
            pd.date_range("2021-01-29", periods=1),
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
            ).to_dict(),
        ]
    else:
        if datetime_loc == "X_index":
            dates = pd.Series(X.index)
        elif datetime_loc == "y_index":
            dates = pd.Series(y.index)
        else:
            dates = X[datetime_column]
        ww_payload = infer_frequency(
            dates,
            debug=True,
            window_length=WINDOW_LENGTH,
            threshold=THRESHOLD,
        )

        col_name = datetime_loc if datetime_loc == 1 else "either index"
        if issue is None:
            assert datetime_format_check.validate(X, y) == []
        elif issue == "missing":
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"Column '{col_name}' has datetime values missing between start and end date.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict(),
                get_uneven_error(col_name, ww_payload),
            ]
        elif issue == "redundant":
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"Column '{col_name}' has more than one row with the same datetime value.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                ).to_dict(),
                get_uneven_error(col_name, ww_payload),
            ]
        else:
            assert datetime_format_check.validate(X, y) == [
                DataCheckError(
                    message=f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals.",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_NO_FREQUENCY_INFERRED,
                ).to_dict(),
            ]


@pytest.mark.parametrize("sort_order", ["increasing", "decreasing", "mixed"])
@pytest.mark.parametrize("datetime_loc", ["datetime_feature", "X_index", "y_index"])
def test_datetime_format_data_check_monotonic(datetime_loc, sort_order):
    X, y = pd.DataFrame({"features": range(300)}), pd.Series(range(300))
    dates = pd.date_range("2021-01-01", periods=300)

    if sort_order == "decreasing":
        dates = dates[::-1]
    elif sort_order == "mixed":
        dates = dates[:50].append(dates[200:]).append(dates[50:200])

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
            message_code=DataCheckMessageCode.DATETIME_NO_FREQUENCY_INFERRED,
        ).to_dict()
        mono_error = DataCheckError(
            message="Datetime values must be sorted in ascending order.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
        ).to_dict()
        if sort_order == "decreasing":
            assert datetime_format_check.validate(X, y) == [mono_error]
        else:
            assert datetime_format_check.validate(X, y) == [mono_error, freq_error]


@pytest.mark.parametrize("n_missing", [2, 5, 7])
def test_datetime_format_data_check_multiple_missing(n_missing):
    X, y = pd.DataFrame({"features": range(100)}), pd.Series(range(100))

    dates = pd.date_range("2021-01-01", periods=15)
    if n_missing == 2:
        # Two missing dates in separate spots
        dates = dates.append(pd.date_range("2021-01-17", periods=86)).drop("2021-01-22")
    elif n_missing == 5:
        # A chunk of 5 missing days in a row
        dates = dates.append(pd.date_range("2021-01-21", periods=85))
    else:
        # Some chunks missing and some alone missing
        dates = dates.append(pd.date_range("2021-01-20", periods=88)).drop("2021-01-27")
        dates = dates.drop("2021-02-22")
        dates = dates.drop("2021-01-11")

    X["dates"] = dates
    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]


def test_datetime_format_data_check_multiple_errors():
    dates = (
        pd.date_range("2021-01-01", periods=9).tolist()
        + ["2021-01-31", "2021-02-02", "2021-02-04"]
        + pd.date_range("2021-02-05", periods=90).tolist()
    )
    X = pd.DataFrame({"dates": dates})
    y = pd.Series(range(21))
    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=9).tolist()
        + ["2021-01-09", "2021-01-31", "2021-02-02", "2021-02-04"]
        + pd.date_range("2021-02-05", periods=90).tolist()
    )
    X = pd.DataFrame({"dates": dates})

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

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
        get_uneven_error("dates", ww_payload),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=15)
        .drop("2021-01-10")
        .append(pd.date_range("2021-01-15", periods=86))
    )
    X = pd.DataFrame({"dates": dates})

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

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
        get_uneven_error("dates", ww_payload),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=15, freq="2D")
        .drop("2021-01-13")
        .append(pd.date_range("2021-01-30", periods=1))
        .append(pd.date_range("2021-01-31", periods=86, freq="2D"))
    )
    X = pd.DataFrame({"dates": dates})

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        DataCheckError(
            message=f"Column 'dates' has datetime values that do not align with the inferred frequency.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_MISALIGNED_VALUES,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]

    dates = (
        pd.date_range("2021-01-01", periods=15, freq="2D")
        .drop("2021-01-13")
        .append(pd.date_range("2021-01-30", periods=1))
        .append(pd.date_range("2021-01-31", periods=86, freq="2D"))
    )
    X = pd.DataFrame({"dates": dates})

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        DataCheckError(
            message=f"Column 'dates' has datetime values that do not align with the inferred frequency.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_MISALIGNED_VALUES,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]


def test_datetime_format_unusual_interval():
    dates = pd.date_range(start="2021-01-01", periods=100, freq="4D")
    X = pd.DataFrame({"dates": dates})
    y = pd.Series(range(100))

    datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")
    assert datetime_format_check.validate(X, y) == []

    dates = dates.drop("2021-01-21")
    X = pd.DataFrame({"dates": dates})
    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    expected = [
        DataCheckError(
            message=f"Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]

    assert datetime_format_check.validate(X, y) == expected

    dates = dates.append(pd.date_range(dates[-1], periods=2, freq="4D"))
    X = pd.DataFrame({"dates": dates})
    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

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
        get_uneven_error("dates", ww_payload),
    ]
    assert datetime_format_check.validate(X, y) == expected


def test_datetime_format_nan_data_check_error():
    dates = pd.Series(pd.date_range(start="2021-01-01", periods=20))
    dates[0] = np.NaN
    X = pd.DataFrame(dates, columns=["date"])

    ww_payload = infer_frequency(
        X["date"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    expected = [
        DataCheckError(
            message="Input datetime column 'date' contains NaN values. Please impute NaN values or drop these rows.",
            data_check_name=DateTimeFormatDataCheck.name,
            message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
        ).to_dict(),
        get_uneven_error("date", ww_payload),
    ]

    dt_nan_check = DateTimeFormatDataCheck(datetime_column="date")
    assert dt_nan_check.validate(X, pd.Series()) == expected

    dates = pd.Series(pd.date_range(start="2021-01-01", periods=100))
    dates[0] = np.NaN
    dates[20] = pd.to_datetime("2021-01-20")
    X = pd.DataFrame(dates, columns=["date"])

    del expected[-1]

    ww_payload = infer_frequency(
        X["date"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

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
            get_uneven_error("date", ww_payload),
        ],
    )
    assert dt_nan_check.validate(X, pd.Series()) == expected


def test_datetime_nan_check_ww():
    dt_nan_check = DateTimeFormatDataCheck(datetime_column="dates")
    y = pd.Series()

    dates = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-08"))
    dates[0] = np.datetime64("NaT")

    ww_input = pd.DataFrame(dates, columns=["dates"])
    ww_input.ww.init()
    ww_payload = infer_frequency(
        ww_input["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    expected = [
        DataCheckError(
            message="Input datetime column 'dates' contains NaN values. Please impute NaN values or drop these rows.",
            data_check_name=DateTimeFormatDataCheck.name,
            message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
        ).to_dict(),
        get_uneven_error("dates", ww_payload),
    ]

    assert dt_nan_check.validate(ww_input, y) == expected
