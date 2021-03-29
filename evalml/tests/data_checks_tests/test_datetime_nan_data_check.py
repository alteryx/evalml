import numpy as np
import pandas as pd
import woodwork as ww

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DateTimeNaNDataCheck
)


def test_datetime_nan_data_check_errors(ts_data):
    data, _ = ts_data
    data.reset_index(inplace=True, drop=False)
    data.at[0, 'index'] = np.NaN
    dt_nan_check = DateTimeNaNDataCheck()
    assert dt_nan_check.validate(data) == {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop this column.',
                                  data_check_name=DateTimeNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                                  details={"columns": 'index'})]
    }


def test_datetime_nan_check_input_formats():
    dt_nan_check = DateTimeNaNDataCheck()

    # test empty pd.DataFrame
    assert dt_nan_check.validate(pd.DataFrame()) == {"warnings": [], "errors": [], "actions": []}

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop this column.',
                                  data_check_name=DateTimeNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                                  details={"columns": 'index'})]
    }

    dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-08'))
    dates[0] = np.datetime64('NaT')

    #  test Woodwork
    ww_input = ww.DataTable(pd.DataFrame(dates, columns=['index']))
    assert dt_nan_check.validate(ww_input) == expected

    expected = {
        "warnings": [],
        "actions": [],
        "errors": [DataCheckError(message='Input datetime column(s) (0) contains NaN values. Please impute NaN values or drop this column.',
                                  data_check_name=DateTimeNaNDataCheck.name,
                                  message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                                  details={'columns': '0'})]
    }

    #  test 2D list
    assert dt_nan_check.validate([dates, np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-08'))]) == expected

    # test np.array
    assert dt_nan_check.validate(np.array([dates, np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-08'))])) == expected
