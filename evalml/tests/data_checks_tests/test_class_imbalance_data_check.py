import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    ClassImbalanceDataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning
)

class_imbalance_data_check_name = ClassImbalanceDataCheck.name


def test_class_imbalance_errors():
    X = pd.DataFrame()

    with pytest.raises(ValueError, match="threshold 0 is not within the range"):
        ClassImbalanceDataCheck(threshold=0).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="threshold 0.51 is not within the range"):
        ClassImbalanceDataCheck(threshold=0.51).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="threshold -0.5 is not within the range"):
        ClassImbalanceDataCheck(threshold=-0.5).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="Provided number of CV folds"):
        ClassImbalanceDataCheck(num_cv_folds=-1).validate(X, y=pd.Series([0, 1, 1]))

    with pytest.raises(ValueError, match="Provided value min_samples"):
        ClassImbalanceDataCheck(min_samples=-1).validate(X, y=pd.Series([0, 1, 1]))


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
def test_class_imbalance_data_check_binary(input_type):
    X = pd.DataFrame()
    y = pd.Series([0, 0, 1])
    y_long = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_balanced = pd.Series([0, 0, 1, 1])

    if input_type == "np":
        X = X.to_numpy()
        y = y.to_numpy()
        y_long = y_long.to_numpy()
        y_balanced = y_balanced.to_numpy()

    elif input_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
        y_long = ww.DataColumn(y_long)
        y_balanced = ww.DataColumn(y_balanced)

    class_imbalance_check = ClassImbalanceDataCheck(min_samples=1, num_cv_folds=0)
    assert class_imbalance_check.validate(X, y) == {"warnings": [], "errors": [], "actions": []}
    assert class_imbalance_check.validate(X, y_long) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict()],
        "errors": [],
        "actions": []
    }
    assert ClassImbalanceDataCheck(threshold=0.25, min_samples=1, num_cv_folds=0).validate(X, y_long) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [1]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [1]}).to_dict()],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_balanced) == {"warnings": [], "errors": [], "actions": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0, 1]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [0, 1]}).to_dict()],
        "actions": []
    }


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
def test_class_imbalance_data_check_multiclass(input_type):
    X = pd.DataFrame()
    y = pd.Series([0, 2, 1, 1])
    y_imbalanced_default_threshold = pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_imbalanced_set_threshold = pd.Series([0, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    y_imbalanced_cv = pd.Series([0, 1, 2, 2, 1, 1, 1])
    y_long = pd.Series([0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4])

    if input_type == "np":
        X = X.to_numpy()
        y = y.to_numpy()
        y_imbalanced_default_threshold = y_imbalanced_default_threshold.to_numpy()
        y_imbalanced_set_threshold = y_imbalanced_set_threshold.to_numpy()
        y_imbalanced_cv = y_imbalanced_cv.to_numpy()
        y_long = y_long.to_numpy()

    elif input_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
        y_imbalanced_default_threshold = ww.DataColumn(y_imbalanced_default_threshold)
        y_imbalanced_set_threshold = ww.DataColumn(y_imbalanced_set_threshold)
        y_imbalanced_cv = ww.DataColumn(y_imbalanced_cv)
        y_long = ww.DataColumn(y_long)

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)
    assert class_imbalance_check.validate(X, y) == {"warnings": [], "errors": [], "actions": []}
    assert class_imbalance_check.validate(X, y_imbalanced_default_threshold) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict(),
                     DataCheckWarning(message="The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": [0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0, min_samples=1).validate(X, y_imbalanced_set_threshold) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [3, 0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [3, 0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=2)
    assert class_imbalance_check.validate(X, y_imbalanced_cv) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [2, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [2, 0]}).to_dict()],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_long) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [1, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [1, 0]}).to_dict()],
        "actions": []
    }

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y_long) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [3, 2, 1, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [3, 2, 1, 0]}).to_dict()],
        "actions": []
    }


@pytest.mark.parametrize("input_type", ["pd", "np", "ww"])
def test_class_imbalance_empty_and_nan(input_type):
    X = pd.DataFrame()
    y_empty = pd.Series([])
    y_has_nan = pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2])

    if input_type == "np":
        X = X.to_numpy()
        y_empty = y_empty.to_numpy()
        y_has_nan = y_has_nan.to_numpy()

    elif input_type == "ww":
        X = ww.DataTable(X)
        y_empty = ww.DataColumn(y_empty)
        y_has_nan = ww.DataColumn(y_has_nan)
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)

    assert class_imbalance_check.validate(X, y_empty) == {"warnings": [], "errors": [], "actions": []}
    assert ClassImbalanceDataCheck(threshold=0.5, min_samples=1, num_cv_folds=0).validate(X, y_has_nan) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 50% of the target: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2.0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=0).validate(X, y_has_nan) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 50% of the target: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2.0]}).to_dict(),
                     DataCheckWarning(message="The following labels in the target have severe class imbalance because they fall under 50% of the target and have less than 100 samples: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": [2.0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y_empty) == {"warnings": [], "errors": [], "actions": []}
    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=1).validate(X, y_has_nan) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 50% of the target: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2.0]}).to_dict(),
                     DataCheckWarning(message="The following labels in the target have severe class imbalance because they fall under 50% of the target and have less than 100 samples: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": [2.0]}).to_dict()],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [2.0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [2.0]}).to_dict()],
        "actions": []
    }


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_class_imbalance_nonnumeric(input_type):
    X = pd.DataFrame()
    y_bools = pd.Series([True, False, False, False, False])
    y_binary = pd.Series(["yes", "no", "yes", "yes", "yes"])
    y_multiclass = pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red", "red", "red"])
    y_multiclass_imbalanced_folds = pd.Series(["No", "Maybe", "Maybe", "No", "Yes"])
    y_binary_imbalanced_folds = pd.Series(["No", "Yes", "No", "Yes", "No"])
    if input_type == "ww":
        X = ww.DataTable(X)
        y_bools = ww.DataColumn(y_bools)
        y_binary = ww.DataColumn(y_binary)
        y_multiclass = ww.DataColumn(y_multiclass)

    class_imbalance_check = ClassImbalanceDataCheck(threshold=0.25, min_samples=1, num_cv_folds=0)
    assert class_imbalance_check.validate(X, y_bools) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [True]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [True]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_binary) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: ['no']",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": ["no"]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert ClassImbalanceDataCheck(threshold=0.35, num_cv_folds=0).validate(X, y_multiclass) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 35% of the target: ['green', 'blue']",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": ["green", "blue"]}).to_dict(),
                     DataCheckWarning(message="The following labels in the target have severe class imbalance because they fall under 35% of the target and have less than 100 samples: ['green', 'blue']",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": ["green", "blue"]}).to_dict()],
        "errors": [],
        "actions": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y_multiclass_imbalanced_folds) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: ['Yes']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["Yes"]}).to_dict()],
        "actions": []
    }
    assert class_imbalance_check.validate(X, y_multiclass) == {"warnings": [], "errors": [], "actions": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y_binary_imbalanced_folds) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["No", "Yes"]}).to_dict()],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_multiclass) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['green', 'blue']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["green", "blue"]}).to_dict()],
        "actions": []
    }


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_class_imbalance_nonnumeric_balanced(input_type):
    X = pd.DataFrame()
    y_bools_balanced = pd.Series([True, True, True, False, False])
    y_binary_balanced = pd.Series(["No", "Yes", "No", "Yes"])
    y_multiclass_balanced = pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"])
    if input_type == "ww":
        X = ww.DataTable(X)
        y_bools_balanced = ww.DataColumn(y_bools_balanced)
        y_binary_balanced = ww.DataColumn(y_binary_balanced)
        y_multiclass_balanced = ww.DataColumn(y_multiclass_balanced)

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y_multiclass_balanced) == {"warnings": [], "errors": [], "actions": []}
    assert class_imbalance_check.validate(X, y_binary_balanced) == {"warnings": [], "errors": [], "actions": []}
    assert class_imbalance_check.validate(X, y_multiclass_balanced) == {"warnings": [], "errors": [], "actions": []}


@pytest.mark.parametrize("input_type", ["pd", "ww"])
@pytest.mark.parametrize("min_samples", [1, 20, 50, 100, 500])
def test_class_imbalance_severe(min_samples, input_type):
    X = pd.DataFrame()
    # 0 will be < 10% of the data, but there will be 50 samples of it
    y_values_binary = pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 50)
    y_values_multiclass = pd.Series([0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] * 50)
    if input_type == "ww":
        X = ww.DataTable(X)
        y_values_binary = ww.DataColumn(y_values_binary)
        y_values_multiclass = ww.DataColumn(y_values_multiclass)

    class_imbalance_check = ClassImbalanceDataCheck(min_samples=min_samples, num_cv_folds=1)
    warnings = [DataCheckWarning(message="The following labels fall below 10% of the target: [0]",
                                 data_check_name=class_imbalance_data_check_name,
                                 message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                 details={"target_values": [0]}).to_dict()]
    if min_samples > 50:
        warnings.append(DataCheckWarning(message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than {min_samples} samples: [0]",
                                         data_check_name=class_imbalance_data_check_name,
                                         message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                         details={"target_values": [0]}).to_dict())
    assert class_imbalance_check.validate(X, y_values_binary) == {
        "warnings": warnings,
        "errors": [],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_values_multiclass) == {
        "warnings": warnings,
        "errors": [],
        "actions": []
    }


def test_class_imbalance_large_multiclass():
    X = pd.DataFrame()
    y_values_multiclass_large = pd.Series([0] * 20 + [1] * 25 + [2] * 99 + [3] * 105 + [4] * 900 + [5] * 900)
    y_multiclass_huge = pd.Series([i % 200 for i in range(100000)])
    y_imbalanced_multiclass_huge = y_multiclass_huge.append(pd.Series([200] * 10), ignore_index=True)
    y_imbalanced_multiclass_nan = y_multiclass_huge.append(pd.Series([np.nan] * 10), ignore_index=True)

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y_values_multiclass_large) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [2, 1, 0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2, 1, 0]}).to_dict(),
                     DataCheckWarning(message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [2, 1, 0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": [2, 1, 0]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_multiclass_huge) == {
        "warnings": [],
        "errors": [],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_imbalanced_multiclass_huge) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [200]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [200]}).to_dict(),
                     DataCheckWarning(message=f"The following labels in the target have severe class imbalance because they fall under 10% of the target and have less than 100 samples: [200]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_SEVERE,
                                      details={"target_values": [200]}).to_dict()],
        "errors": [],
        "actions": []
    }

    assert class_imbalance_check.validate(X, y_imbalanced_multiclass_nan) == {
        "warnings": [],
        "errors": [],
        "actions": []
    }
