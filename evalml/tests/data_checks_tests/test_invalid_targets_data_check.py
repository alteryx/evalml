import numpy as np
import pandas as pd

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    InvalidTargetDataCheck
)
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)

invalid_targets_data_check_name = InvalidTargetDataCheck.name


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("regression")

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == {"warnings": [], "errors": []}
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (100.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL).to_dict()]
    }


def test_invalid_target_data_check_numeric_binary_classification_valid_float():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([0.0, 1.0, 0.0, 1.0])) == {"warnings": [], "errors": []}


def test_invalid_target_data_check_numeric_binary_classification_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([1, 5, 1, 5, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError(message="Numerical binary classification target classes must be [0, 1], got [1, 5] instead",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES).to_dict()]
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 5, np.nan, np.nan])) == {
        "warnings": [],
        "errors": [DataCheckError(message="2 row(s) (50.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL).to_dict(),
                   DataCheckError(message="Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES).to_dict()]
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 1, 1, 0, 1, 2])) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }


def test_invalid_target_data_check_invalid_data_types_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    valid_data_types = numeric_and_boolean_dtypes + categorical_dtypes
    y = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
    for data_type in valid_data_types:
        y = y.astype(data_type)
        assert invalid_targets_check.validate(X, y) == {"warnings": [], "errors": []}

    y = pd.date_range('2000-02-03', periods=5, freq='W')
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_data_types)),
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck("binary")
    X = pd.DataFrame()

    # test None
    messages = invalid_targets_check.validate(X, y=None)
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }

    #  test list
    messages = invalid_targets_check.validate(X, [None, None, None, 0])
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (75.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }

    # test np.array
    messages = invalid_targets_check.validate(X, np.array([None, None, None, 0]))
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (75.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES).to_dict()]
    }
