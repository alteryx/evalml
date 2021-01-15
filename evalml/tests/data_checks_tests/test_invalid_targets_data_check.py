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
from evalml.utils.gen_utils import numeric_and_boolean_ww

invalid_targets_data_check_name = InvalidTargetDataCheck.name


def test_invalid_target_data_check_invalid_n_unique():
    with pytest.raises(ValueError, match="`n_unique` must be a non-negative integer value."):
        InvalidTargetDataCheck("regression", get_default_primary_search_objective("regression"), n_unique=-1)


def test_invalid_target_data_check_nan_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("regression", get_default_primary_search_objective("regression"))

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
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    assert invalid_targets_check.validate(X, y=pd.Series([0.0, 1.0, 0.0, 1.0])) == {"warnings": [], "errors": []}


def test_invalid_target_data_check_numeric_binary_classification_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    assert invalid_targets_check.validate(X, y=pd.Series([1, 5, 1, 5, 1, 1])) == {
        "warnings": [DataCheckWarning(
            message="Numerical binary classification target classes must be [0, 1], got [1, 5] instead",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
            details={"target_values": [1, 5]}).to_dict()],
        "errors": []
    }
    assert invalid_targets_check.validate(X, y=pd.Series([0, 5, np.nan, np.nan])) == {
        "warnings": [DataCheckWarning(
            message="Numerical binary classification target classes must be [0, 1], got [5.0, 0.0] instead",
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
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [1, 0, 2]}).to_dict()]
    }


def test_invalid_target_data_check_multiclass_two_examples_per_class():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("multiclass", get_default_primary_search_objective("binary"))
    expected_message = "Target does not have at least two instances per class which is required for multiclass classification"

    # with 1 class not having min 2 instances
    assert invalid_targets_check.validate(X, y=pd.Series([0] + [1] * 19 + [2] * 80)) == {
        "warnings": [],
        "errors": [DataCheckError(message=expected_message,
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_EXAMPLES_PER_CLASS,
                                  details={"least_populated_class_labels": [0]}).to_dict()]
    }
    # with 2 classes not having min 2 instances
    assert invalid_targets_check.validate(X, y=pd.Series([0] + [1] + [2] * 98)) == {
        "warnings": [],
        "errors": [DataCheckError(message=expected_message,
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_EXAMPLES_PER_CLASS,
                                  details={"least_populated_class_labels": [1, 0]}).to_dict()]
    }


@pytest.mark.parametrize("pd_type", ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool'])
def test_invalid_target_data_check_invalid_pandas_data_types_error(pd_type):
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    y = pd.Series([0, 1, 0, 0, 1, 0, 1, 0])
    y = y.astype(pd_type)
    assert invalid_targets_check.validate(X, y) == {"warnings": [], "errors": []}

    y = pd.Series(pd.date_range('2000-02-03', periods=5, freq='W'))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Target is unsupported {} type. Valid Woodwork logical types include: {}"
                                  .format("Datetime",
                                          ", ".join([ltype.type_string for ltype in numeric_and_boolean_ww])),
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                                  details={"unsupported_type": "datetime"}).to_dict(),
                   DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }


def test_invalid_target_y_none():
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    with pytest.raises(ValueError, match="y cannot be None"):
        invalid_targets_check.validate(pd.DataFrame(), y=None)


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))
    X = pd.DataFrame()

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == {
        "warnings": [],
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
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
                   DataCheckError(message="Binary class targets require exactly two unique values.",
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
                   DataCheckError(message="Binary class targets require exactly two unique values.",
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
                   DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": [0]}).to_dict()]
    }


def test_invalid_target_data_check_n_unique():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"))

    # Test default value of n_unique
    y = pd.Series(list(range(100, 200)) + list(range(200)))
    unique_values = y.value_counts().index.tolist()[:100]  # n_unique defaults to 100
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }

    # Test number of unique values < n_unique
    y = pd.Series(range(20))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }

    # Test n_unique is None
    invalid_targets_check = InvalidTargetDataCheck("binary", get_default_primary_search_objective("binary"),
                                                   n_unique=None)
    y = pd.Series(range(150))
    unique_values = y.value_counts().index.tolist()
    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="Binary class targets require exactly two unique values.",
                                  data_check_name=invalid_targets_data_check_name,
                                  message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                  details={"target_values": unique_values}).to_dict()]
    }


@pytest.mark.parametrize("objective",
                         ['Root Mean Squared Log Error', 'Mean Squared Log Error', 'Mean Absolute Percentage Error'])
def test_invalid_target_data_check_invalid_labels_for_nonnegative_objective_names(objective):
    X = pd.DataFrame({'column_one': [100, 200, 100, 200, 200, 100, 200, 100] * 25})
    y = pd.Series([2, 2, 3, 3, -1, -1, 1, 1] * 25)

    data_checks = DataChecks([InvalidTargetDataCheck], {"InvalidTargetDataCheck": {"problem_type": "multiclass",
                                                                                   "objective": objective}})
    assert data_checks.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(
            message=f"Target has non-positive values which is not supported for {objective}",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
            details={"Count of offending values": sum(val <= 0 for val in y.values.flatten())}).to_dict()]
    }

    X = pd.DataFrame({'column_one': [100, 200, 100, 200, 100]})
    y = pd.Series([2, 3, 0, 1, 1])

    invalid_targets_check = InvalidTargetDataCheck(problem_type="regression", objective=objective)

    assert invalid_targets_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(
            message=f"Target has non-positive values which is not supported for {objective}",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
            details={"Count of offending values": sum(val <= 0 for val in y.values.flatten())}).to_dict()]
    }


@pytest.mark.parametrize("objective", [RootMeanSquaredLogError(), MeanSquaredLogError(), MAPE()])
def test_invalid_target_data_check_invalid_labels_for_nonnegative_objective_instances(objective):
    X = pd.DataFrame({'column_one': [100, 200, 100, 200, 200, 100, 200, 100] * 25})
    y = pd.Series([2, 2, 3, 3, -1, -1, 1, 1] * 25)

    data_checks = DataChecks([InvalidTargetDataCheck], {"InvalidTargetDataCheck": {"problem_type": "multiclass",
                                                                                   "objective": objective}})

    assert data_checks.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(
            message=f"Target has non-positive values which is not supported for {objective.name}",
            data_check_name=invalid_targets_data_check_name,
            message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
            details={"Count of offending values": sum(val <= 0 for val in y.values.flatten())}).to_dict()]
    }


def test_invalid_target_data_check_invalid_labels_for_objectives(time_series_core_objectives):
    X = pd.DataFrame({'column_one': [100, 200, 100, 200, 200, 100, 200, 100] * 25})
    y = pd.Series([2, 2, 3, 3, -1, -1, 1, 1] * 25)

    for objective in time_series_core_objectives:
        if not objective.positive_only:
            data_checks = DataChecks([InvalidTargetDataCheck], {"InvalidTargetDataCheck": {"problem_type": "multiclass",
                                                                                           "objective": objective}})
            assert data_checks.validate(X, y) == {
                "warnings": [],
                "errors": []
            }

    X = pd.DataFrame({'column_one': [100, 200, 100, 200, 100]})
    y = pd.Series([2, 3, 0, 1, 1])

    for objective in time_series_core_objectives:
        if not objective.positive_only:
            invalid_targets_check = InvalidTargetDataCheck(problem_type="regression", objective=objective)
            assert invalid_targets_check.validate(X, y) == {
                "warnings": [],
                "errors": []
            }


@pytest.mark.parametrize("objective",
                         ['Root Mean Squared Log Error', 'Mean Squared Log Error', 'Mean Absolute Percentage Error'])
def test_invalid_target_data_check_valid_labels_for_nonnegative_objectives(objective):
    X = pd.DataFrame({'column_one': [100, 100, 200, 300, 100, 200, 100] * 25})
    y = pd.Series([2, 2, 3, 3, 1, 1, 1] * 25)

    data_checks = DataChecks([InvalidTargetDataCheck], {"InvalidTargetDataCheck": {"problem_type": "multiclass",
                                                                                   "objective": objective}})
    assert data_checks.validate(X, y) == {
        "warnings": [],
        "errors": []
    }


def test_invalid_target_data_check_initialize_with_none_objective():
    with pytest.raises(DataCheckInitError, match="Encountered the following error"):
        DataChecks([InvalidTargetDataCheck], {"InvalidTargetDataCheck": {"problem_type": "multiclass",
                                                                         "objective": None}})


@pytest.mark.parametrize("problem_type",
                         ['regression'])
def test_invalid_target_data_check_regression_problem_nonnumeric_data(problem_type):
    X = pd.DataFrame()
    y_categorical = pd.Series(["Peace", "Is", "A", "Lie"] * 100)
    y_mixed_cat_numeric = pd.Series(["Peace", 2, "A", 4] * 100)
    y_integer = pd.Series([1, 2, 3, 4])
    y_float = pd.Series([1.1, 2.2, 3.3, 4.4])
    y_numeric = pd.Series([1, 2.2, 3, 4.4])

    data_check_error = DataCheckError(
        message=f"Target data type should be numeric for regression type problems.",
        data_check_name=invalid_targets_data_check_name,
        message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
        details={}).to_dict()

    invalid_targets_check = InvalidTargetDataCheck(problem_type, get_default_primary_search_objective(problem_type))
    assert invalid_targets_check.validate(X, y=y_categorical) == {"warnings": [], "errors": [data_check_error]}
    assert invalid_targets_check.validate(X, y=y_mixed_cat_numeric) == {"warnings": [], "errors": [data_check_error]}
    assert invalid_targets_check.validate(X, y=y_integer) == {"warnings": [], "errors": []}
    assert invalid_targets_check.validate(X, y=y_float) == {"warnings": [], "errors": []}
    assert invalid_targets_check.validate(X, y=y_numeric) == {"warnings": [], "errors": []}


def test_invalid_target_data_check_multiclass_problem_binary_data():
    X = pd.DataFrame()
    y_multiclass = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3] * 25)
    y_binary = pd.Series([0, 1, 1, 1, 0, 0] * 25)

    data_check_error = DataCheckError(
        message=f"Target has two or less classes, which is too few for multiclass problems.  Consider changing to binary.",
        data_check_name=invalid_targets_data_check_name,
        message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_ENOUGH_CLASSES,
        details={"num_classes": len(set(y_binary))}).to_dict()

    invalid_targets_check = InvalidTargetDataCheck("multiclass", get_default_primary_search_objective("multiclass"))
    assert invalid_targets_check.validate(X, y=y_multiclass) == {"warnings": [], "errors": []}
    assert invalid_targets_check.validate(X, y=y_binary) == {"warnings": [], "errors": [data_check_error]}


def test_invalid_target_data_check_multiclass_problem_almostcontinuous_data():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck("multiclass", get_default_primary_search_objective("multiclass"))

    y_multiclass_high_classes = pd.Series(list(range(0, 100)) * 3)  # 100 classes, 300 samples, .33 class/sample ratio
    data_check_error = DataCheckWarning(
        message=f"Target has a large number of unique values, could be regression type problem.",
        data_check_name=invalid_targets_data_check_name,
        message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
        details={"class_to_value_ratio": 1 / 3}).to_dict()
    assert invalid_targets_check.validate(X, y=y_multiclass_high_classes) == {"warnings": [data_check_error],
                                                                              "errors": []}

    y_multiclass_med_classes = pd.Series(list(range(0, 5)) * 20)  # 5 classes, 100 samples, .05 class/sample ratio
    data_check_error = DataCheckWarning(
        message=f"Target has a large number of unique values, could be regression type problem.",
        data_check_name=invalid_targets_data_check_name,
        message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
        details={"class_to_value_ratio": .05}).to_dict()
    assert invalid_targets_check.validate(X, y=y_multiclass_med_classes) == {"warnings": [data_check_error],
                                                                             "errors": []}

    y_multiclass_low_classes = pd.Series(list(range(0, 3)) * 100)  # 2 classes, 300 samples, .01 class/sample ratio
    assert invalid_targets_check.validate(X, y=y_multiclass_low_classes) == {"warnings": [], "errors": []}
