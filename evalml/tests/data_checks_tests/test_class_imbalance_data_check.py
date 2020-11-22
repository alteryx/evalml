import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import DataCheckError, DataCheckWarning
from evalml.data_checks.class_imbalance_data_check import (
    ClassImbalanceDataCheck
)


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
        "warnings": [DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }
    assert ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0).validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning("The following labels fall below 25% of the target: [0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [0]", "ClassImbalanceDataCheck").to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series([0, 0, 1, 1])) == {"warnings": [], "errors": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [1, 0]", "ClassImbalanceDataCheck").to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series(['No', 'No', 'Yes'])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']", "ClassImbalanceDataCheck").to_dict()]
    }


def test_class_imbalance_data_check_multiclass():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 1, 1])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    assert ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0).validate(X, y=pd.Series([0, 2, 2, 2, 3, 3, 1, 1, 1, 1])) == {
        "warnings": [DataCheckWarning("The following labels fall below 25% of the target: [3, 0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=2)
    assert class_imbalance_check.validate(X, y=pd.Series([0, 0, 1, 2, 2, 1, 1, 1])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [2, 0]", "ClassImbalanceDataCheck").to_dict()]
    }

    y = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
    assert class_imbalance_check.validate(X, y=pd.Series(y)) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 4 instances: [1, 0]", "ClassImbalanceDataCheck").to_dict()]
    }

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series(y)) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [3, 2, 1, 0]", "ClassImbalanceDataCheck").to_dict()]
    }


def test_class_imbalance_empty_and_nan():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=pd.Series([])) == {"warnings": [], "errors": []}
    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=0).validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2])) == {
        "warnings": [DataCheckWarning("The following labels fall below 50% of the target: [2.0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series([])) == {"warnings": [], "errors": []}
    assert ClassImbalanceDataCheck(threshold=0.5, num_cv_folds=1).validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2])) == {
        "warnings": [DataCheckWarning("The following labels fall below 50% of the target: [2.0]", "ClassImbalanceDataCheck").to_dict()],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: [2.0]", "ClassImbalanceDataCheck").to_dict()]
    }


def test_class_imbalance_nonnumeric():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck(threshold=0.25, num_cv_folds=0)

    assert class_imbalance_check.validate(X, y=[True, False, False, False, False]) == {
        "warnings": [DataCheckWarning("The following labels fall below 25% of the target: [True]", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    assert class_imbalance_check.validate(X, y=["yes", "no", "yes", "yes", "yes"]) == {
        "warnings": [DataCheckWarning("The following labels fall below 25% of the target: ['no']", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    assert ClassImbalanceDataCheck(threshold=0.35, num_cv_folds=0).validate(X, y=["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"]) == {
        "warnings": [DataCheckWarning("The following labels fall below 35% of the target: ['green', 'blue']", "ClassImbalanceDataCheck").to_dict()],
        "errors": []
    }

    class_imbalance_check = ClassImbalanceDataCheck(num_cv_folds=1)
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Yes", "No", "Yes"])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series([True, True, True, False, False])) == {"warnings": [], "errors": []}
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Maybe", "Maybe", "No", "Yes"])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 2 instances: ['Yes']", "ClassImbalanceDataCheck").to_dict()]
    }
    assert class_imbalance_check.validate(X, y=pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"])) == {"warnings": [], "errors": []}

    class_imbalance_check = ClassImbalanceDataCheck()
    assert class_imbalance_check.validate(X, y=pd.Series(["No", "Yes", "No", "Yes", "No"])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['No', 'Yes']", "ClassImbalanceDataCheck").to_dict()]
    }

    assert class_imbalance_check.validate(X, y=pd.Series(["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"])) == {
        "warnings": [],
        "errors": [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: ['red', 'green', 'blue']", "ClassImbalanceDataCheck").to_dict()]
    }
