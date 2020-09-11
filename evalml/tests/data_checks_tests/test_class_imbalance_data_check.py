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
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("The following labels fall below 25% of the target: [0]", "ClassImbalanceDataCheck")]


def test_class_imbalance_data_check_multiclass():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 1, 1])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("The following labels fall below 10% of the target: [0]", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 2, 3, 3, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("The following labels fall below 25% of the target: [3, 0]", "ClassImbalanceDataCheck")]


def test_class_imbalance_empty_and_nan():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 1, 1, 1, 2]), threshold=0.5) == [DataCheckWarning("The following labels fall below 50% of the target: [2.0]", "ClassImbalanceDataCheck")]


def test_class_imbalance_nonnumeric():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=[True, False, False, False, False], threshold=0.25) == [DataCheckWarning("The following labels fall below 25% of the target: [True]", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=["yes", "no", "yes", "yes", "yes"], threshold=0.25) == [DataCheckWarning("The following labels fall below 25% of the target: ['no']", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=["red", "green", "red", "red", "blue", "green", "red", "blue", "green", "red"], threshold=0.35) == [DataCheckWarning("The following labels fall below 35% of the target: ['green', 'blue']", "ClassImbalanceDataCheck")]
