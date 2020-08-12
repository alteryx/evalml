import numpy as np
import pandas as pd

from evalml.data_checks import DataCheckError
from evalml.data_checks.invalid_targets_data_check import (
    InvalidTargetDataCheck
)
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck()

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == []
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == [DataCheckError("3 row(s) (100.0%) of target values are null", "InvalidTargetDataCheck")]


def test_invalid_target_data_check_numeric_binary_classification_valid_float():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck()
    assert invalid_targets_check.validate(X, y=pd.Series([0.0, 1.0, 0.0, 1.0])) == []


def test_invalid_target_data_check_numeric_binary_classification_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck()
    assert invalid_targets_check.validate(X, y=pd.Series([1, 5, 1, 5, 1, 1])) == [DataCheckError("Numerical binary classification target classes must be [0, 1], got [1, 5] instead", "InvalidTargetDataCheck")]
    assert invalid_targets_check.validate(X, y=pd.Series([0, 5, np.nan, np.nan])) == [DataCheckError("2 row(s) (50.0%) of target values are null", "InvalidTargetDataCheck"),
                                                                                      DataCheckError("Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead", "InvalidTargetDataCheck")]


def test_invalid_target_data_check_invalid_data_types_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck()
    valid_data_types = numeric_and_boolean_dtypes + categorical_dtypes
    y = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
    for data_type in valid_data_types:
        y = y.astype(data_type)
        assert invalid_targets_check.validate(X, y) == []

    y = pd.date_range('2000-02-03', periods=5, freq='W')
    assert invalid_targets_check.validate(X, y) == [DataCheckError("Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_data_types)), "InvalidTargetDataCheck")]


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck()
    X = pd.DataFrame()

    # test None
    messages = invalid_targets_check.validate(X, y=None)
    assert messages == []

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == []

    #  test list
    messages = invalid_targets_check.validate(X, [None, None, None, 0])
    assert messages == [DataCheckError("3 row(s) (75.0%) of target values are null", "InvalidTargetDataCheck")]

    # test np.array
    messages = invalid_targets_check.validate(X, np.array([None, None, None, 0]))
    assert messages == [DataCheckError("3 row(s) (75.0%) of target values are null", "InvalidTargetDataCheck")]
