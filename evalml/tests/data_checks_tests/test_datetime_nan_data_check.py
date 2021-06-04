import numpy as np
import pandas as pd

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DateTimeNaNDataCheck,
)


def test_datetime_nan_data_check_error(ts_data):
    data, _ = ts_data
    data.reset_index(inplace=True, drop=False)
    data.at[0, "index"] = np.NaN
    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": [
            DataCheckError(
                message="Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop these rows or columns.",
                data_check_name=DateTimeNaNDataCheck.name,
                message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                details={"columns": "index"},
            ).to_dict()
        ],
    }


def test_datetime_nan_data_check_error_numeric_columns_no_null():
    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(
        pd.DataFrame(np.random.randint(0, 10, size=(10, 4)))
    ) == {"warnings": [], "actions": [], "errors": []}


def test_datetime_nan_data_check_error_numeric_null_columns():
    data = pd.DataFrame(np.random.randint(0, 10, size=(10, 4)))
    data = data.replace(data.iloc[0][0], None)
    data = data.replace(data.iloc[1][1], None)
    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(data) == {"warnings": [], "actions": [], "errors": []}


def test_datetime_nan_data_check_multiple_dt_no_nan():
    data = pd.DataFrame()
    data["A"] = pd.Series(pd.date_range("20200101", periods=3))
    data["B"] = pd.Series(pd.date_range("20200101", periods=3))
    data["C"] = np.random.randint(0, 5, size=len(data))

    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(data) == {"warnings": [], "actions": [], "errors": []}


def test_datetime_nan_data_check_multiple_nan_dt():
    data = pd.DataFrame()
    data["A"] = pd.Series(pd.date_range("20200101", periods=3))
    data.loc[0][0] = None
    data["B"] = pd.Series(pd.date_range("20200101", periods=3))
    data.loc[0][1] = None
    data["C"] = np.random.randint(0, 5, size=len(data))

    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": [
            DataCheckError(
                message="Input datetime column(s) (A, B) contains NaN values. Please impute NaN values or drop these rows or columns.",
                data_check_name=DateTimeNaNDataCheck.name,
                message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                details={"columns": "A, B"},
            ).to_dict()
        ],
    }


def test_datetime_nan_check_input_formats():
    dt_nan_check = DateTimeNaNDataCheck()

    # test empty pd.DataFrame
    assert dt_nan_check.validate(pd.DataFrame()) == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [
            DataCheckError(
                message="Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop these rows or columns.",
                data_check_name=DateTimeNaNDataCheck.name,
                message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                details={"columns": "index"},
            ).to_dict()
        ],
    }

    dates = np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-08"))
    dates[0] = np.datetime64("NaT")

    #  test Woodwork
    ww_input = pd.DataFrame(dates, columns=["index"])
    ww_input.ww.init()
    assert dt_nan_check.validate(ww_input) == expected

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [
            DataCheckError(
                message="Input datetime column(s) (0) contains NaN values. Please impute NaN values or drop these rows or columns.",
                data_check_name=DateTimeNaNDataCheck.name,
                message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                details={"columns": "0"},
            ).to_dict()
        ],
    }

    #  test 2D list
    assert (
        dt_nan_check.validate(
            [dates, np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-08"))]
        )
        == expected
    )

    # test np.array
    assert (
        dt_nan_check.validate(
            np.array(
                [
                    dates,
                    np.arange(np.datetime64("2017-01-01"), np.datetime64("2017-01-08")),
                ]
            )
        )
        == expected
    )
