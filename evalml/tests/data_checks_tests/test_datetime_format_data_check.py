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


def get_uneven_error(col_name, ww_payload, series=None):
    series_message = f"A frequency was detected in column '{col_name}' for series '{series}', but there are faulty datetime values that need to be addressed."
    error = DataCheckError(
        message=f"A frequency was detected in column '{col_name}', but there are faulty datetime values that need to be addressed."
        if series is None
        else series_message,
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
@pytest.mark.parametrize(
    "datetime_loc, is_multiseries, repeat",
    [
        (1, True, 2),
        (1, False, 1),
        ("X_index", True, 2),
        ("X_index", False, 1),
        ("y_index", False, 1),
    ],
)
def test_datetime_format_data_check_typeerror_uneven_intervals(
    issue,
    input_type,
    datetime_loc,
    is_multiseries,
    repeat,
):
    if is_multiseries:
        time_length = 60
    else:
        time_length = 30

    X, y = pd.DataFrame({"features": range(time_length)}), pd.Series(range(time_length))
    if is_multiseries:
        X["series_id"] = pd.Series(list(range(2)) * 30, dtype="str")

    if issue == "type_errors":
        dates = range(time_length)
    else:
        dates = pd.date_range("2021-01-01", periods=time_length)

    if issue == "missing":
        # Skips 2021-01-25 and starts again at 2021-01-27, skipping a date and triggering the error
        dates = (pd.date_range("2021-01-01", periods=25).repeat(repeat)).append(
            (pd.date_range("2021-01-27", periods=5).repeat(repeat)),
        )
    if issue == "uneven":
        dates_1 = pd.date_range("2015-01-01", periods=5, freq="D").repeat(repeat)
        dates_2 = pd.date_range("2015-01-08", periods=3, freq="D").repeat(repeat)
        dates_3 = pd.DatetimeIndex(["2015-01-12"]).repeat(repeat)
        dates_4 = pd.date_range("2015-01-15", periods=5, freq="D").repeat(repeat)
        dates_5 = pd.date_range("2015-01-22", periods=5, freq="D").repeat(repeat)
        dates_6 = pd.date_range("2015-01-29", periods=11, freq="M").repeat(repeat)

        dates = (
            dates_1.append(dates_2)
            .append(dates_3)
            .append(dates_4)
            .append(dates_5)
            .append(dates_6)
        )
    if issue == "redundant":
        dates = (pd.date_range("2021-01-01", periods=29).repeat(repeat)).append(
            (pd.date_range("2021-01-29", periods=1).repeat(repeat)),
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

    if is_multiseries:
        datetime_format_check = DateTimeFormatDataCheck(
            datetime_column=datetime_column,
            series_id="series_id",
        )
    else:
        datetime_format_check = DateTimeFormatDataCheck(datetime_column=datetime_column)

    all_series = X["series_id"].unique() if is_multiseries else [0]
    messages = []

    for series in all_series:
        if issue == "type_errors":
            if len(messages) == 0:
                # type error only gives 1 message regardless of how many series there are
                messages.append(
                    DataCheckError(
                        message="Datetime information could not be found in the data, or was not in a supported datetime format.",
                        data_check_name=datetime_format_check_name,
                        message_code=DataCheckMessageCode.DATETIME_INFORMATION_NOT_FOUND,
                    ).to_dict(),
                )
        else:
            if is_multiseries:
                curr_series_df = X[X[datetime_format_check.series_id] == series]

            # separates the datetimes so it only displays the dates that correspond to the current series
            if input_type == "ww" and is_multiseries:
                # ww makes the series_id into ints so need to cast series into ints
                if datetime_loc == "X_index":
                    dates = pd.Series(
                        X[X[datetime_format_check.series_id] == int(series)].index,
                    )
                else:
                    dates = X[X[datetime_format_check.series_id] == int(series)][
                        datetime_column
                    ]
            elif datetime_loc == "X_index":
                if is_multiseries:
                    dates = pd.Series(curr_series_df.index)
                else:
                    dates = pd.Series(X.index)
            elif datetime_loc == "y_index":
                dates = pd.Series(y.index)
            else:
                if is_multiseries:
                    dates = pd.Series(curr_series_df[datetime_column])
                else:
                    dates = X[datetime_column]
            ww_payload_expected = infer_frequency(
                # this part might cause issues
                dates.reset_index(drop=True),
                debug=True,
                window_length=WINDOW_LENGTH,
                threshold=THRESHOLD,
            )

            col_name = datetime_loc if datetime_loc == 1 else "either index"
            if issue is None:
                break
            elif issue == "missing":
                if is_multiseries:
                    message = f"Column '{col_name}' for series '{series}' has datetime values missing between start and end date."
                    uneven_error = get_uneven_error(
                        col_name,
                        ww_payload_expected,
                        series,
                    )
                else:
                    message = f"Column '{col_name}' has datetime values missing between start and end date."
                    uneven_error = get_uneven_error(col_name, ww_payload_expected)
                messages.extend(
                    [
                        DataCheckError(
                            message=message,
                            data_check_name=datetime_format_check_name,
                            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                        ).to_dict(),
                        uneven_error,
                    ],
                )
            elif issue == "redundant":
                if is_multiseries:
                    message = f"Column '{col_name}' for series '{series}' has more than one row with the same datetime value."
                    uneven_error = get_uneven_error(
                        col_name,
                        ww_payload_expected,
                        series,
                    )
                else:
                    message = f"Column '{col_name}' has more than one row with the same datetime value."
                    uneven_error = get_uneven_error(col_name, ww_payload_expected)
                messages.extend(
                    [
                        DataCheckError(
                            message=message,
                            data_check_name=datetime_format_check_name,
                            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                        ).to_dict(),
                        uneven_error,
                    ],
                )
            else:
                if is_multiseries:
                    message = f"No frequency could be detected in column '{col_name}' for series '{series}', possibly due to uneven intervals or too many duplicate/missing values."
                else:
                    message = f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals or too many duplicate/missing values."

                messages.append(
                    DataCheckError(
                        message=message,
                        data_check_name=datetime_format_check_name,
                        message_code=DataCheckMessageCode.DATETIME_NO_FREQUENCY_INFERRED,
                    ).to_dict(),
                )
    if issue is None:
        assert datetime_format_check.validate(X, y) == []
    else:
        assert datetime_format_check.validate(X, y) == messages


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
            message=f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals or too many duplicate/missing values.",
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
@pytest.mark.parametrize("is_multiseries, repeat", [(True, 2), (False, 1)])
def test_datetime_format_data_check_multiple_missing(n_missing, is_multiseries, repeat):
    X, y = pd.DataFrame({"features": range(100)}), pd.Series(range(100))
    if is_multiseries:
        X["series_id"] = pd.Series(list(range(2)) * 50, dtype="str")

    dates = pd.date_range("2021-01-01", periods=15).repeat(repeat)
    if n_missing == 2:
        # Two missing dates in separate spots
        if is_multiseries:
            dates = dates.append(
                pd.date_range("2021-01-17", periods=36).repeat(2),
            ).drop(
                "2021-01-22",
            )
        else:
            dates = dates.append(pd.date_range("2021-01-17", periods=86)).drop(
                "2021-01-22",
            )
    elif n_missing == 5:
        # A chunk of 5 missing days in a row
        if is_multiseries:
            dates = dates.append(pd.date_range("2021-01-21", periods=35).repeat(2))
        else:
            dates = dates.append(pd.date_range("2021-01-21", periods=85))
    else:
        # Some chunks missing and some alone missing
        if is_multiseries:
            dates = dates.append(
                pd.date_range("2021-01-19", periods=39).repeat(2),
            ).drop(
                "2021-01-27",
            )
            dates = dates.drop("2021-01-20")
        else:
            dates = dates.append(pd.date_range("2021-01-20", periods=88)).drop(
                "2021-01-27",
            )
        dates = dates.drop("2021-02-22")
        dates = dates.drop("2021-01-11")

    X["dates"] = dates

    if is_multiseries:
        datetime_format_check = DateTimeFormatDataCheck(
            datetime_column="dates",
            series_id="series_id",
        )
    else:
        datetime_format_check = DateTimeFormatDataCheck(datetime_column="dates")

    messages = []
    series_list = X["series_id"].unique() if is_multiseries else [0]

    for series in series_list:
        observed_ts = (
            X[X["series_id"] == series]["dates"].reset_index(drop=True)
            if is_multiseries
            else X["dates"]
        )
        ww_payload_expected = infer_frequency(
            observed_ts,
            debug=True,
            window_length=WINDOW_LENGTH,
            threshold=THRESHOLD,
        )
        if is_multiseries:
            message = f"""Column 'dates' for series '{series}' has datetime values missing between start and end date."""
            uneven_error = get_uneven_error("dates", ww_payload_expected, series)
        else:
            message = """Column 'dates' has datetime values missing between start and end date."""
            uneven_error = get_uneven_error("dates", ww_payload_expected)
        messages.extend(
            [
                DataCheckError(
                    message=message,
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict(),
                uneven_error,
            ],
        )
    assert datetime_format_check.validate(X, y) == messages


def test_datetime_format_data_check_multiple_errors():
    dates = (
        pd.date_range("2021-01-01", periods=9).tolist()
        + ["2021-01-31", "2021-02-02", "2021-02-04"]
        + pd.date_range("2021-02-05", periods=90).tolist()
    )
    X = pd.DataFrame({"dates": dates}, dtype="datetime64[ns]")
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
            message="Column 'dates' has datetime values missing between start and end date.",
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
    X = pd.DataFrame({"dates": dates}, dtype="datetime64[ns]")

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message="Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message="Column 'dates' has datetime values missing between start and end date.",
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
    X = pd.DataFrame({"dates": dates}, dtype="datetime64[ns]")

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message="Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message="Column 'dates' has datetime values missing between start and end date.",
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
    X = pd.DataFrame({"dates": dates}, dtype="datetime64[ns]")

    ww_payload = infer_frequency(
        X["dates"],
        debug=True,
        window_length=WINDOW_LENGTH,
        threshold=THRESHOLD,
    )

    assert datetime_format_check.validate(X, y) == [
        DataCheckError(
            message="Column 'dates' has datetime values missing between start and end date.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
        ).to_dict(),
        DataCheckError(
            message="Column 'dates' has datetime values that do not align with the inferred frequency.",
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
            message="Column 'dates' has datetime values missing between start and end date.",
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
            message="Column 'dates' has more than one row with the same datetime value.",
            data_check_name=datetime_format_check_name,
            message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
        ).to_dict(),
        DataCheckError(
            message="Column 'dates' has datetime values missing between start and end date.",
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
                message="Column 'date' has more than one row with the same datetime value.",
                data_check_name=datetime_format_check_name,
                message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
            ).to_dict(),
            DataCheckError(
                message="Column 'date' has datetime values missing between start and end date.",
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


def test_datetime_many_duplicates_and_nans():
    dates = pd.Series(pd.date_range(start="1/1/2021", periods=76))
    nans = pd.Series([None] * 12)
    duplicates = pd.Series(pd.date_range(start="1/1/2021", periods=12))
    dates = pd.concat([dates, nans, duplicates])

    X = pd.DataFrame({"date": dates}, columns=["date"])
    X = X.reset_index(drop=True)
    y = pd.Series(range(len(dates)))

    dc = DateTimeFormatDataCheck(datetime_column="date")
    result = dc.validate(X, y)

    assert result[2]["code"] == "DATETIME_HAS_UNEVEN_INTERVALS"

    X.iloc[-25, 0] = None
    dc = DateTimeFormatDataCheck(datetime_column="date", nan_duplicate_threshold=0.70)
    result = dc.validate(X, y)

    assert result[2]["code"] == "DATETIME_HAS_UNEVEN_INTERVALS"

    dc = DateTimeFormatDataCheck(datetime_column="date")
    result = dc.validate(X, y)

    assert result[2]["code"] == "DATETIME_NO_FREQUENCY_INFERRED"


def test_datetime_format_data_check_invalid_seriesid_multiseries(
    multiseries_ts_data_stacked,
):
    X, y = multiseries_ts_data_stacked
    datetime_format_check = DateTimeFormatDataCheck(
        datetime_column="Date",
        series_id="not_series_id",
    )
    with pytest.raises(
        ValueError,
        match="""series_id "not_series_id" is not in the dataset.""",
    ):
        datetime_format_check.validate(X, y)


@pytest.mark.parametrize("nans", [0, 1, 2])
def test_datetime_format_data_check_nan_multiseries(nans):
    dates = pd.Series(pd.date_range(start="2021-01-01", periods=20).repeat(2))
    if nans == 1:
        dates[0] = np.NaN
    elif nans == 2:
        dates[0] = np.NaN
        dates[1] = np.NaN
    X = pd.DataFrame(dates, columns=["date"])
    X["series_id"] = pd.Series(list(range(2)) * 20, dtype="str")

    messages = []
    for series in X["series_id"].unique():
        ww_payload_expected = infer_frequency(
            X[X["series_id"] == series]["date"].reset_index(drop=True),
            debug=True,
            window_length=WINDOW_LENGTH,
            threshold=THRESHOLD,
        )
        if (series == "0" and nans >= 1) or (series == "1" and nans >= 2):
            messages.extend(
                [
                    DataCheckError(
                        message=f"""Input datetime column 'date' for series '{series}' contains NaN values. Please impute NaN values or drop these rows.""",
                        data_check_name=DateTimeFormatDataCheck.name,
                        message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                    ).to_dict(),
                    get_uneven_error("date", ww_payload_expected, series),
                ],
            )

    dt_nan_check = DateTimeFormatDataCheck(
        datetime_column="date",
        series_id="series_id",
    )
    assert dt_nan_check.validate(X, pd.Series(dtype="float64")) == messages


def test_datetime_format_data_check_multiseries_not_aligned_frequency():
    dates = (
        pd.date_range("2021-01-01", periods=15, freq="2D")
        .repeat(2)
        .drop("2021-01-13")
        .append(pd.date_range("2021-01-30", periods=1).repeat(2))
        .append(pd.date_range("2021-01-31", periods=35, freq="2D").repeat(2))
    )
    X = pd.DataFrame({"dates": dates}, dtype="datetime64[ns]")
    X["series_id"] = pd.Series(list(range(2)) * 50, dtype="str")
    datetime_format_check = DateTimeFormatDataCheck(
        datetime_column="dates",
        series_id="series_id",
    )

    messages = []
    for series in X["series_id"].unique():
        ww_payload_expected = infer_frequency(
            X[X["series_id"] == series]["dates"].reset_index(drop=True),
            debug=True,
            window_length=WINDOW_LENGTH,
            threshold=THRESHOLD,
        )

        messages.extend(
            [
                DataCheckError(
                    message=f"""Column 'dates' for series '{series}' has datetime values missing between start and end date.""",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict(),
                DataCheckError(
                    message=f"""Column 'dates' for series '{series}' has datetime values that do not align with the inferred frequency.""",
                    data_check_name=datetime_format_check_name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_MISALIGNED_VALUES,
                ).to_dict(),
                get_uneven_error("dates", ww_payload_expected, series),
            ],
        )
    assert datetime_format_check.validate(X, pd.Series(dtype="float64")) == messages
