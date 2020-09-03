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

    with pytest.raises(ValueError, match="threshold 1 is not within the range"):
        class_imbalance_check.validate(X, y=pd.Series([0, 1, 1]), threshold=1)

    with pytest.raises(ValueError, match="threshold -0.5 is not within the range"):
        class_imbalance_check.validate(X, y=pd.Series([0, 1, 1]), threshold=-0.5)


def test_class_imbalance_data_check_binary():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("Label 0 makes up 9.091% of the target data, which is below the acceptable threshold of 10%", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("Label 0 makes up 9.091% of the target data, which is below the acceptable threshold of 25%", "ClassImbalanceDataCheck")]


def test_class_imbalance_data_check_multiclass():
    X = pd.DataFrame()
    class_imbalance_check = ClassImbalanceDataCheck()

    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 1, 1])) == []
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])) == [DataCheckWarning("Label 0 makes up 7.692% of the target data, which is below the acceptable threshold of 10%", "ClassImbalanceDataCheck")]
    assert class_imbalance_check.validate(X, y=pd.Series([0, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), threshold=0.25) == [DataCheckWarning("Label 2 makes up 23.529% of the target data, which is below the acceptable threshold of 25%", "ClassImbalanceDataCheck"),
                                                                                                                                   DataCheckWarning("Label 3 makes up 11.765% of the target data, which is below the acceptable threshold of 25%", "ClassImbalanceDataCheck"),
                                                                                                                                   DataCheckWarning("Label 0 makes up 5.882% of the target data, which is below the acceptable threshold of 25%", "ClassImbalanceDataCheck")]
