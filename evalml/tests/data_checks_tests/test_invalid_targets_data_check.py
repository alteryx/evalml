import numpy as np
import pandas as pd
import pytest

from evalml.automl import get_default_primary_search_objective
from evalml.data_checks import (
    DataCheckError,
    DataCheckMessageCode,
    DataChecks,
    DataCheckWarning,
    InvalidTargetDataCheck
)
from evalml.exceptions import DataCheckInitError
from evalml.objectives import (
    MAPE,
    MeanSquaredLogError,
    RootMeanSquaredLogError
)
from evalml.utils.woodwork_utils import numeric_and_boolean_ww

invalid_targets_data_check_name = InvalidTargetDataCheck.name


def test_invalid_target_data_check_invalid_n_unique():
    with pytest.raises(ValueError, match="`n_unique` must be a non-negative integer value."):
        InvalidTargetDataCheck("regression", get_default_primary_search_objective("regression"), n_unique=-1)


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame({"col": [1, 2, 3]})
    invalid_targets_check = InvalidTargetDataCheck("regression", get_default_primary_search_objective("regression"))

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == {"warnings": [], "errors": [], "actions": []}
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == {
        "warnings": [],
        "errors": [DataCheckError(message="3 row(s) (100.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 3, "pct_null_rows": 100}).to_dict()],
        "actions": []
    }


def test_invalid_target_data_check_numeric_binary_classification_valid_float():
    y = pd.Series([0.0, 1.0, 0.0, 1.0])
    X = pd.DataFrame({"col": range(len(y))})
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    assert invalid_targets_check.validate(X, y) == {"warnings": [], "errors": [], "actions": []}


def test_invalid_target_data_check_numeric_binary_classification_error():
    y = pd.Series([1, 5, 1, 5, 1, 1])
    X = pd.DataFrame({"col": range(len(y))})
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [DataCheckWarning(
            message="Numerical binary classification target classes must be [0, 1], got [1, 5] instead",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
            details={"target_values": [1, 5]}).to_dict()],
        "errors": [],
        "actions": []
    }

    y = pd.Series([0, 5, np.nan, np.nan])
    X = pd.DataFrame({"col": range(len(y))})
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [DataCheckWarning(
            message="Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
            details={"target_values": [5.0, 0.0]}).to_dict()],
        "errors": [DataCheckError(message="2 row(s) (50.0%) of target values are null",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                  details={"num_null_rows": 2, "pct_null_rows": 50}).to_dict()],
        "actions": []
    }

    y = pd.Series([0, 1, 1, 0, 1, 2])
    X = pd.DataFrame({"col": range(len(y))})
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [1, 0, 2]}).to_dict()],
        "actions": []
    }


def test_invalid_target_data_check_multiclass_two_examples_per_class():
    y = pd.Series([0] + [1] * 19 + [2] * 80)
    X = pd.DataFrame({"col": range(len(y))})
    invalid_targets_check = InvalidTargetDataCheck("multiclass", get_default_primary_search_objective("binary"))
    expected_message = "Target does not have at least two instances per class which is required for multiclass classification"

    # with 1 class not having min 2 instances
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message=expected_message,
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_EXAMPLES_PER_CLASS,
                                  details={"least_populated_class_labels": [0]}).to_dict()],
        "actions": []
    }

    y = pd.Series([0] + [1] + [2] * 98)
    X = pd.DataFrame({"col": range(len(y))})
    # with 2 classes not having min 2 instances
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message=expected_message,
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_EXAMPLES_PER_CLASS,
                                  details={"least_populated_class_labels": [1, 0]}).to_dict()],
        "actions": []
    }
