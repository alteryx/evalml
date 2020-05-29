import pandas as pd
import pytest

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.label_leakage_data_check import LabelLeakageDataCheck


def test_label_leakage_data_check_init():
    label_leakage_check = LabelLeakageDataCheck()
    assert label_leakage_check.pct_corr_threshold == 0.95

    label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.0)
    assert label_leakage_check.pct_corr_threshold == 0

    label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.5)
    assert label_leakage_check.pct_corr_threshold == 0.5

    label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=1.0)
    assert label_leakage_check.pct_corr_threshold == 1.0

    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        LabelLeakageDataCheck(pct_corr_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        LabelLeakageDataCheck(pct_corr_threshold=1.1)


def test_label_leakage_data_check_warnings():
    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.5)
    assert label_leakage_check.validate(X, y) == [DataCheckWarning("Column 'a' is 50.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                  DataCheckWarning("Column 'b' is 50.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                  DataCheckWarning("Column 'c' is 50.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                  DataCheckWarning("Column 'd' is 50.0% or more correlated with the target", "LabelLeakageDataCheck")]


def test_label_leakage_data_check_input_formats():
    label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame, empty pd.Series
    assert label_leakage_check.validate(pd.DataFrame(), pd.Series()) == []

    y = pd.Series([1, 0, 1, 1])
    X = pd.DataFrame()
    X["a"] = y * 3
    X["b"] = y - 1
    X["c"] = y / 10
    X["d"] = ~y
    X["e"] = [0, 0, 0, 0]
    y = y.astype(bool)

    expected_messages = [DataCheckWarning("Column 'a' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                         DataCheckWarning("Column 'b' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                         DataCheckWarning("Column 'c' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                         DataCheckWarning("Column 'd' is 80.0% or more correlated with the target", "LabelLeakageDataCheck")]

    #  test y as list
    assert label_leakage_check.validate(X, y.values) == expected_messages

    # test X as np.array
    assert label_leakage_check.validate(X.to_numpy(), y) == [DataCheckWarning("Column '0' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                             DataCheckWarning("Column '1' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                             DataCheckWarning("Column '2' is 80.0% or more correlated with the target", "LabelLeakageDataCheck"),
                                                             DataCheckWarning("Column '3' is 80.0% or more correlated with the target", "LabelLeakageDataCheck")]
