import numpy as np
import pandas as pd

from evalml.data_checks import DataCheckError, DataCheckMessageType
from evalml.data_checks.invalid_targets_data_check import (
    InvalidTargetDataCheck
)
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("regression")

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == {DataCheckMessageType.WARNING: [], DataCheckMessageType.ERROR: []}
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("3 row(s) (100.0%) of target values are null", "InvalidTargetDataCheck")]
    }


def test_invalid_target_data_check_numeric_binary_classification_valid_float():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([0.0, 1.0, 0.0, 1.0])) == {DataCheckMessageType.WARNING: [], DataCheckMessageType.ERROR: []}


def test_invalid_target_data_check_numeric_binary_classification_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([1, 5, 1, 5, 1, 1])) == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("Numerical binary classification target classes must be [0, 1], got [1, 5] instead", "InvalidTargetDataCheck")]
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 5, np.nan, np.nan])) == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("2 row(s) (50.0%) of target values are null", "InvalidTargetDataCheck"),
                                     DataCheckError("Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead", "InvalidTargetDataCheck")]
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 1, 1, 0, 1, 2])) == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }


def test_invalid_target_data_check_invalid_data_types_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    valid_data_types = numeric_and_boolean_dtypes + categorical_dtypes
    y = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
    for data_type in valid_data_types:
        y = y.astype(data_type)
        assert invalid_targets_check.validate(X, y) == {DataCheckMessageType.WARNING: [], DataCheckMessageType.ERROR: []}

    y = pd.date_range('2000-02-03', periods=5, freq='W')
    assert invalid_targets_check.validate(X, y) == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_data_types)), "InvalidTargetDataCheck"),
                                     DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck("binary")
    X = pd.DataFrame()

    # test None
    messages = invalid_targets_check.validate(X, y=None)
    assert messages == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }

    #  test list
    messages = invalid_targets_check.validate(X, [None, None, None, 0])
    assert messages == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("3 row(s) (75.0%) of target values are null", "InvalidTargetDataCheck"),
                                     DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }

    # test np.array
    messages = invalid_targets_check.validate(X, np.array([None, None, None, 0]))
    assert messages == {
        DataCheckMessageType.WARNING: [],
        DataCheckMessageType.ERROR: [DataCheckError("3 row(s) (75.0%) of target values are null", "InvalidTargetDataCheck"),
                                     DataCheckError("Target does not have two unique values which is not supported for binary classification", "InvalidTargetDataCheck")]
    }
