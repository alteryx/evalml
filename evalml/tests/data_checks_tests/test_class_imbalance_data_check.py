import numpy as np
import pandas as pd
import pytest

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


def test_class_imbalance_data_check_binary():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)
    assert class_imbalance_check.validate(X, y=[0, 1, 1]) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict()],
        "errors": []
    }
    assert ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0).validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [0]}).to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series([0, 0, 1, 1])) == {"warnings": [], "errors": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [1, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [1, 0]}).to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series(['No', 'No', 'Yes'])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["No", "Yes"]}).to_dict()]
    }


def test_class_imbalance_data_check_multiclass():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 1, 1])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 10% of the target: [0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [0]}).to_dict()],
        "errors": []
    }

    assert ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0).validate(X, y=pd.Series([0, 2, 2, 2, 3, 3, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [3, 0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [3, 0]}).to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=2)
    assert class_imbalance_check.validate(X, y=pd.Series([0, 0, 1, 2, 2, 1, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [2, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [2, 0]}).to_dict()]
    }

    y = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    assert class_imbalance_check.validate(X, y=pd.Series(y)) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [1, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [1, 0]}).to_dict()]
    }

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series(y)) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [3, 2, 1, 0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [3, 2, 1, 0]}).to_dict()]
    }


def test_class_imbalance_empty_and_nan():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=pd.Series([])) == {"warnings": [], "errors": []}
    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=0).validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 50% of the target: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2.0]}).to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series([])) == {"warnings": [], "errors": []}
    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=1).validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2])) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 50% of the target: [2.0]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [2.0]}).to_dict()],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [2.0]",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": [2.0]}).to_dict()]
    }


def test_class_imbalance_nonnumeric():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=[True, False, False, False, False]) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: [True]",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": [True]}).to_dict()],
        "errors": []
    }

    assert class_imbalance_check.validate(X, y=["yes", "no", "yes", "yes", "yes"]) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 25% of the target: ['no']",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": ["no"]}).to_dict()],
        "errors": []
    }

    assert ClassImbalanceDataCheck(threshold=0.35, num_cv_folds=0).validate(X, y=["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"]) == {
        "warnings": [DataCheckWarning(message="The following labels fall below 35% of the target: ['green', 'blue']",
                                      data_check_name=class_imbalance_data_check_name,
                                      message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_THRESHOLD,
                                      details={"target_values": ["green", "blue"]}).to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Yes", "No", "Yes"])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([True, True, True, False, False])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Maybe", "Maybe", "No", "Yes"])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: ['Yes']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["Yes"]}).to_dict()]
    }
    assert class_imbalance_check.validate(X, y=pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"])) == {"warnings": [], "errors": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Yes", "No", "Yes", "No"])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["No", "Yes"]}).to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"])) == {
        "warnings": [],
        "errors": [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['red', 'green', 'blue']",
                                  data_check_name=class_imbalance_data_check_name,
                                  message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                  details={"target_values": ["red", "green", "blue"]}).to_dict()]
    }
