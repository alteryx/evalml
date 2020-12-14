import string

import numpy as np
import pandas as pd
import woodwork as ww

from evalml.data_checks import (
    DataCheckMessageCode,
    DataCheckWarning,
    OutliersDataCheck
)
from evalml.utils import get_random_state

outliers_data_check_name = OutliersDataCheck.name


def test_outliers_data_check_init():
    outliers_check = OutliersDataCheck()
    assert outliers_check.random_state.get_state()[0] == get_random_state(0).get_state()[0]

    outliers_check = OutliersDataCheck(random_state=2)
    assert outliers_check.random_state.get_state()[0] == get_random_state(2).get_state()[0]


def test_outliers_data_check_warnings():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000
    X.iloc[:, 90] = 'string_values'

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": []
    }


def test_outliers_data_check_input_formats():
    outliers_check = OutliersDataCheck()

    # test empty pd.DataFrame
    assert outliers_check.validate(pd.DataFrame()) == {"warnings": [], "errors": []}

    # test np.array
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X.to_numpy()) == {
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": []
    }

    # test Woodwork
    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(ww.DataTable(X)) == {
        "warnings": [DataCheckWarning(message="Column(s) '3', '25', '55', '72' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": [3, 25, 55, 72]}).to_dict()],
        "errors": []
    }


def test_outliers_data_check_string_cols():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 2))
    n_cols = 20

    X = pd.DataFrame(data=data, columns=[string.ascii_lowercase[i] for i in range(n_cols)])
    X.iloc[0, 3] = 1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X) == {
        "warnings": [DataCheckWarning(message="Column(s) 'd' are likely to have outlier data.",
                                      data_check_name=outliers_data_check_name,
                                      message_code=DataCheckMessageCode.HAS_OUTLIERS,
                                      details={"columns": ["d"]}).to_dict()],
        "errors": []
    }
