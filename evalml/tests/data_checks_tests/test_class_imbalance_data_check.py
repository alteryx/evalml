import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import DataCheckWarning
from evalml.data_checks.class_imbalance_data_check import (
    ClassImbalanceDataCheck
)


def test_class_imbalance_invalid_threshold():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    with pytest.raises(ValueError, match="threshold 0 is not within the range"):
        class_imbalance_check.validate(X, y=pd.Series([0, 1, 1]), threshold=0)

    with pytest.raises(ValueError, match="threshold 0.51 is not within the range"):
        class_imbalance_check.validate(X, y=pd.Series([0, 1, 1]), threshold=0.51)

    with pytest.raises(ValueError, match="threshold -0.5 is not within the range"):
        class_imbalance_check.validate(X, y=pd.Series([0, 1, 1]), threshold=-0.5)


def test_class_imbalance_data_check_binary():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=[0, 1, 1]) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("Label '0' makes up 9.09% of the target data, which is below the recommended threshold of 10%", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("Label '0' makes up 9.09% of the target data, which is below the recommended threshold of 25%", "ClassImbalanceDataCheck")]


def test_class_imbalance_data_check_multiclass():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 1, 1])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("Label '0' makes up 7.69% of the target data, which is below the recommended threshold of 10%", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 2, 3, 3, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("Label '3' makes up 20.00% of the target data, which is below the recommended threshold of 25%", "ClassImbalanceDataCheck"),
                                                                                                              DataCheckWarning("Label '0' makes up 10.00% of the target data, which is below the recommended threshold of 25%", "ClassImbalanceDataCheck")]


def test_class_imbalance_empty_and_nan():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2]), threshold=0.5) == [DataCheckWarning("Label '2.0' makes up 20.00% of the target data, which is below the recommended threshold of 50%", "ClassImbalanceDataCheck")]


def test_class_imbalance_nonnumeric():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=[True, False, False, False, False], threshold=0.25) == [DataCheckWarning("Label 'True' makes up 20.00% of the target data, which is below the recommended threshold of 25%", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=["yes", "no", "yes", "yes", "yes"], threshold=0.25) == [DataCheckWarning("Label 'no' makes up 20.00% of the target data, which is below the recommended threshold of 25%", "ClassImbalanceDataCheck")]
    assert sorted(class_imbalance_check.validate(X, y=["red", "blue", "green", "red", "blue", "green", "red", "blue", "green", "red"], threshold=0.35), key=lambda x: x.message) == [DataCheckWarning("Label 'blue' makes up 30.00% of the target data, which is below the recommended threshold of 35%", "ClassImbalanceDataCheck"),
                                                                                                                                                                                     DataCheckWarning("Label 'green' makes up 30.00% of the target data, which is below the recommended threshold of 35%", "ClassImbalanceDataCheck")]
