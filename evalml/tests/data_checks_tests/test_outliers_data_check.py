import numpy as np
import pandas as pd

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.outliers_data_check import OutliersDataCheck
from evalml.utils import get_random_state


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
    assert outliers_check.validate(X) == [DataCheckWarning("Column '3' is likely to have outlier data", "OutliersDataCheck"),
                                          DataCheckWarning("Column '25' is likely to have outlier data", "OutliersDataCheck"),
                                          DataCheckWarning("Column '55' is likely to have outlier data", "OutliersDataCheck"),
                                          DataCheckWarning("Column '72' is likely to have outlier data", "OutliersDataCheck")]


def test_outliers_data_check_input_formats():
    outliers_check = OutliersDataCheck()

    # test empty pd.DataFrame
    assert outliers_check.validate(pd.DataFrame()) == []

    # test np.array
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[0, 3] = 1000
    X.iloc[3, 25] = 1000
    X.iloc[5, 55] = 10000
    X.iloc[10, 72] = -1000

    outliers_check = OutliersDataCheck()
    assert outliers_check.validate(X.to_numpy()) == [DataCheckWarning("Column '3' is likely to have outlier data", "OutliersDataCheck"),
                                                     DataCheckWarning("Column '25' is likely to have outlier data", "OutliersDataCheck"),
                                                     DataCheckWarning("Column '55' is likely to have outlier data", "OutliersDataCheck"),
                                                     DataCheckWarning("Column '72' is likely to have outlier data", "OutliersDataCheck")]
