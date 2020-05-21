import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.detect_id_columns_data_check import (
    DetectIDColumnsDataCheck
)


def test_id_cols_data_check_init():
    id_cols_check = DetectIDColumnsDataCheck()
    assert id_cols_check.pct_null_threshold == 0.95

    id_cols_check = DetectIDColumnsDataCheck(pct_null_threshold=0.0)
    assert id_cols_check.pct_null_threshold == 0

    id_cols_check = DetectIDColumnsDataCheck(pct_null_threshold=0.5)
    assert id_cols_check.pct_null_threshold == 0.5

    id_cols_check = DetectIDColumnsDataCheck(pct_null_threshold=1.0)
    assert id_cols_check.pct_null_threshold == 1.0

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DetectIDColumnsDataCheck(pct_null_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        DetectIDColumnsDataCheck(pct_null_threshold=1.1)


def test_id_cols_data_check_warnings():
    data = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                         'all_null': [None, None, None, None, None],
                         'no_null': [1, 2, 3, 4, 5]})
    no_null_check = DetectIDColumnsDataCheck(pct_null_threshold=0.0)
    assert no_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is more than 0% null", "DetectIDColumnsDataCheck"),
                                            DataCheckWarning("Column 'all_null' is more than 0% null", "DetectIDColumnsDataCheck")]
    some_null_check = DetectIDColumnsDataCheck(pct_null_threshold=0.5)
    assert some_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is 50.0% or more null", "DetectIDColumnsDataCheck"),
                                              DataCheckWarning("Column 'all_null' is 50.0% or more null", "DetectIDColumnsDataCheck")]
    all_null_check = DetectIDColumnsDataCheck(pct_null_threshold=1.0)
    assert all_null_check.validate(data) == [DataCheckWarning("Column 'all_null' is 100.0% or more null", "DetectIDColumnsDataCheck")]


def test_id_cols_data_check_input_formats():
    id_cols_check = DetectIDColumnsDataCheck(pct_null_threshold=0.8)

    # test empty pd.DataFrame
    messages = id_cols_check.validate(pd.DataFrame())
    assert messages == []

    #  test list
    messages = id_cols_check.validate([None, None, None, None, 5])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectIDColumnsDataCheck")]

    #  test pd.Series
    messages = id_cols_check.validate(pd.Series([None, None, None, None, 5]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectIDColumnsDataCheck")]

    #  test 2D list
    messages = id_cols_check.validate([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectIDColumnsDataCheck")]

    # test np.array
    messages = id_cols_check.validate(np.array([[None, None, None, None, 0], [None, None, None, "hi", 5]]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectIDColumnsDataCheck")]
