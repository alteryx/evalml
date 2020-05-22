import numpy as np
import pandas as pd

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.detect_outliers_data_check import (
    DetectOutliersDataCheck
)
from evalml.utils import get_random_state


def test_outliers_data_check_init():
    outliers_check = DetectOutliersDataCheck()
    assert outliers_check.random_state.get_state()[0] == get_random_state(0).get_state()[0]

    outliers_check = DetectOutliersDataCheck(random_state=2)
    assert outliers_check.random_state.get_state()[0] == get_random_state(2).get_state()[0]


def test_outliers_data_check_warnings():
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[3, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[25, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[55, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[72, :] = pd.Series(np.random.randn(100) * 1000)

    outliers_check = DetectOutliersDataCheck()
    assert outliers_check.validate(X) == [DataCheckWarning("Row '3' is likely to have outlier data", "DetectOutliersDataCheck"),
                                          DataCheckWarning("Row '25' is likely to have outlier data", "DetectOutliersDataCheck"),
                                          DataCheckWarning("Row '55' is likely to have outlier data", "DetectOutliersDataCheck"),
                                          DataCheckWarning("Row '72' is likely to have outlier data", "DetectOutliersDataCheck")]


def test_outliers_data_check_input_formats():
    outliers_check = DetectOutliersDataCheck()

    # test empty pd.DataFrame
    messages = outliers_check.validate(pd.DataFrame())
    assert messages == []

    # test np.array
    a = np.arange(10) * 0.01
    data = np.tile(a, (100, 10))

    X = pd.DataFrame(data=data)
    X.iloc[3, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[25, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[55, :] = pd.Series(np.random.randn(100) * 1000)
    X.iloc[72, :] = pd.Series(np.random.randn(100) * 1000)

    outliers_check = DetectOutliersDataCheck()
    assert outliers_check.validate(X.to_numpy()) == [DataCheckWarning("Row '3' is likely to have outlier data", "DetectOutliersDataCheck"),
                                                     DataCheckWarning("Row '25' is likely to have outlier data", "DetectOutliersDataCheck"),
                                                     DataCheckWarning("Row '55' is likely to have outlier data", "DetectOutliersDataCheck"),
                                                     DataCheckWarning("Row '72' is likely to have outlier data", "DetectOutliersDataCheck")]
