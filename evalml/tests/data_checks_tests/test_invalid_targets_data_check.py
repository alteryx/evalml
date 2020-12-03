import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
    InvalidTargetDataCheck
)
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)

invalid_targets_data_check_name = InvalidTargetDataCheck.name


def test_invalid_target_data_check_invalid_n_unique():
    with pytest.raises(ValueError, match="`n_unique` must be a non-negative integer value."):
        InvalidTargetDataCheck("regression", n_unique=-1)


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("regression")

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == {"warnings": [], "errors": []}
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (100.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 3, "pct_null_rows": 100}).to_dict()]
    }


def test_invalid_target_data_check_numeric_binary_classification_valid_float():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([0.0, 1.0, 0.0, 1.0])) == {"warnings": [], "errors": []}


def test_invalid_target_data_check_numeric_binary_classification_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    assert invalid_targets_check.validate(X, y=pd.Series([1, 5, 1, 5, 1, 1])) == {
        "warnings": [DataCheckWarning(message="Numerical binary classification target classes must be [0, 1], got [1, 5] instead",
                                      data_check_name=invalid_targets_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
                                      details={"target_values": [1, 5]}).to_dict()],
        "errors": []
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 5, np.nan, np.nan])) == {
        "warnings": [DataCheckWarning(message="Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead",
                                      data_check_name=invalid_targets_data_check_name,
                                      message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
                                      details={"target_values": [5.0, 0.0]}).to_dict()],
        "errors": [DataCheckError(message="2 row(s) (50.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 2, "pct_null_rows": 50}).to_dict()]
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 1, 1, 0, 1, 2])) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [1, 0, 2]}).to_dict()]
    }


def test_invalid_target_data_check_invalid_data_types_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")
    valid_data_types = numeric_and_boolean_dtypes + categorical_dtypes
    y = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
    for data_type in valid_data_types:
        y = y.astype(data_type)
        assert invalid_targets_check.validate(X, y) == {"warnings": [], "errors": []}

    y = pd.Series(pd.date_range('2000-02-03', periods=5, freq='W'))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_data_types)),
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                                  details={"unsupported_type": y.dtype}).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }


def test_invalid_target_y_none():
    invalid_targets_check = InvalidTargetDataCheck("binary")
    with pytest.raises(ValueError, match="y cannot be None"):
        invalid_targets_check.validate(pd.DataFrame(), y=None)


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck("binary")
    X = pd.DataFrame()

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": []}).to_dict()]
    }
    #  test Woodwork
    messages = invalid_targets_check.validate(X, pd.Series([None, None, None, 0]))
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (75.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 3, "pct_null_rows": 75}).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [0]}).to_dict()]
    }

    #  test list
    messages = invalid_targets_check.validate(X, [None, None, None, 0])
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (75.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 3, "pct_null_rows": 75}).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [0]}).to_dict()]
    }

    # test np.array
    messages = invalid_targets_check.validate(X, np.array([None, None, None, 0]))
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (75.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 3, "pct_null_rows": 75}).to_dict(),
                   DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [0]}).to_dict()]
    }


def test_invalid_target_data_check_n_unique():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary")

    # Test default value of n_unique
    y = pd.Series(list(range(100, 200)) + list(range(200)))
    unique_values = y.value_counts().index.tolist()[:100]  # n_unique defaults to 100
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }

    # Test number of unique values < n_unique
    y = pd.Series(range(20))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }

    # Test n_unique is None
    invalid_targets_check = InvalidTargetDataCheck("binary", n_unique=None)
    y = pd.Series(range(150))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }
