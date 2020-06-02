import numpy as np
import pandas as pd

from evalml.data_checks import DataCheckError
from evalml.data_checks.invalid_targets_data_check import (
    InvalidTargetDataCheck
)


def test_invalid_target_data_check_error():
    X = pd.DataFrame()
    invalid_targets_check = InvalidTargetDataCheck()

    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, 3])) == []
    assert invalid_targets_check.validate(X, y=pd.Series([1, 2, np.nan])) == [DataCheckError("Row(s) 2 contains a null value", "InvalidTargetDataCheck")]
    assert invalid_targets_check.validate(X, y=pd.Series([np.nan, np.nan, np.nan])) == [DataCheckError("Row(s) 0, 1, 2 contains a null value", "InvalidTargetDataCheck")]


def test_invalid_target_data_input_formats():
    invalid_targets_check = InvalidTargetDataCheck()
    X = pd.DataFrame()

    # test None
    messages = invalid_targets_check.validate(X, y=None)
    assert messages == []

    # test empty pd.Series
    messages = invalid_targets_check.validate(X, pd.Series())
    assert messages == []

    #  test list
    messages = invalid_targets_check.validate(X, [None, None, None, 0])
    assert messages == [DataCheckError("Row(s) 0, 1, 2 contains a null value", "InvalidTargetDataCheck")]

    # test np.array
    messages = invalid_targets_check.validate(X, np.array([None, None, None, 0]))
    assert messages == [DataCheckError("Row(s) 0, 1, 2 contains a null value", "InvalidTargetDataCheck")]
