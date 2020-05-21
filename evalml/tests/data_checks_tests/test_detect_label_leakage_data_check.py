import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.detect_label_leakage_data_check import (
    DetectLabelLeakageDataCheck
)


def test_label_leakage_data_check_init():
    label_leakage_check = DetectLabelLeakageDataCheck()
    assert label_leakage_check.pct_corr_threshold == 0.95

    label_leakage_check = DetectLabelLeakageDataCheck(pct_corr_threshold=0.0)
    assert label_leakage_check.pct_corr_threshold == 0

    label_leakage_check = DetectLabelLeakageDataCheck(pct_corr_threshold=0.5)
    assert label_leakage_check.pct_corr_threshold == 0.5

    label_leakage_check = DetectLabelLeakageDataCheck(pct_corr_threshold=1.0)
    assert label_leakage_check.pct_corr_threshold == 1.0

    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        DetectLabelLeakageDataCheck(pct_corr_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_corr_threshold must be a float between 0 and 1, inclusive."):
        DetectLabelLeakageDataCheck(pct_corr_threshold=1.1)


def test_label_leakage_data_check_warnings():
    data = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                         'all_null': [None, None, None, None, None],
                         'no_null': [1, 2, 3, 4, 5]})
    no_null_check = DetectLabelLeakageDataCheck(pct_corr_threshold=0.0)
    assert no_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is more than 0% null", "DetectLabelLeakageDataCheck"),
                                            DataCheckWarning("Column 'all_null' is more than 0% null", "DetectLabelLeakageDataCheck")]
    some_null_check = DetectLabelLeakageDataCheck(pct_corr_threshold=0.5)
    assert some_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is 50.0% or more null", "DetectLabelLeakageDataCheck"),
                                              DataCheckWarning("Column 'all_null' is 50.0% or more null", "DetectLabelLeakageDataCheck")]
    all_null_check = DetectLabelLeakageDataCheck(pct_corr_threshold=1.0)
    assert all_null_check.validate(data) == [DataCheckWarning("Column 'all_null' is 100.0% or more null", "DetectLabelLeakageDataCheck")]


def test_label_leakage_data_check_input_formats():
    label_leakage_check = DetectLabelLeakageDataCheck(pct_corr_threshold=0.8)

    # test empty pd.DataFrame
    messages = label_leakage_check.validate(pd.DataFrame())
    assert messages == []

    #  test list
    messages = label_leakage_check.validate([None, None, None, None, 5])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectLabelLeakageDataCheck")]

    #  test pd.Series
    messages = label_leakage_check.validate(pd.Series([None, None, None, None, 5]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectLabelLeakageDataCheck")]

    #  test 2D list
    messages = label_leakage_check.validate([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectLabelLeakageDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectLabelLeakageDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectLabelLeakageDataCheck")]

    # test np.array
    messages = label_leakage_check.validate(np.array([[None, None, None, None, 0], [None, None, None, "hi", 5]]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectLabelLeakageDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectLabelLeakageDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectLabelLeakageDataCheck")]
