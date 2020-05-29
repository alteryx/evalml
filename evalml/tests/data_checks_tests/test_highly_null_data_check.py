import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.highly_null_data_check import HighlyNullDataCheck


def test_highly_null_data_check_init():
    highly_null_check = HighlyNullDataCheck()
    assert highly_null_check.pct_null_threshold == 0.95

    highly_null_check = HighlyNullDataCheck(pct_null_threshold=0.0)
    assert highly_null_check.pct_null_threshold == 0

    highly_null_check = HighlyNullDataCheck(pct_null_threshold=0.5)
    assert highly_null_check.pct_null_threshold == 0.5

    highly_null_check = HighlyNullDataCheck(pct_null_threshold=1.0)
    assert highly_null_check.pct_null_threshold == 1.0

    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        HighlyNullDataCheck(pct_null_threshold=-0.1)
    with pytest.raises(ValueError, match="pct_null_threshold must be a float between 0 and 1, inclusive."):
        HighlyNullDataCheck(pct_null_threshold=1.1)


def test_highly_null_data_check_warnings():
    data = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                         'all_null': [None, None, None, None, None],
                         'no_null': [1, 2, 3, 4, 5]})
    no_null_check = HighlyNullDataCheck(pct_null_threshold=0.0)
    assert no_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is more than 0% null", "HighlyNullDataCheck"),
                                            DataCheckWarning("Column 'all_null' is more than 0% null", "HighlyNullDataCheck")]
    some_null_check = HighlyNullDataCheck(pct_null_threshold=0.5)
    assert some_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is 50.0% or more null", "HighlyNullDataCheck"),
                                              DataCheckWarning("Column 'all_null' is 50.0% or more null", "HighlyNullDataCheck")]
    all_null_check = HighlyNullDataCheck(pct_null_threshold=1.0)
    assert all_null_check.validate(data) == [DataCheckWarning("Column 'all_null' is 100.0% or more null", "HighlyNullDataCheck")]


def test_highly_null_data_check_input_formats():
    highly_null_check = HighlyNullDataCheck(pct_null_threshold=0.8)

    # test empty pd.DataFrame
    assert highly_null_check.validate(pd.DataFrame()) == []

    #  test list
    assert highly_null_check.validate([None, None, None, None, 5]) == [DataCheckWarning("Column '0' is 80.0% or more null", "HighlyNullDataCheck")]

    #  test pd.Series
    assert highly_null_check.validate(pd.Series([None, None, None, None, 5])) == [DataCheckWarning("Column '0' is 80.0% or more null", "HighlyNullDataCheck")]

    #  test 2D list
    assert highly_null_check.validate([[None, None, None, None, 0], [None, None, None, "hi", 5]]) == [DataCheckWarning("Column '0' is 80.0% or more null", "HighlyNullDataCheck"),
                                                                                                      DataCheckWarning("Column '1' is 80.0% or more null", "HighlyNullDataCheck"),
                                                                                                      DataCheckWarning("Column '2' is 80.0% or more null", "HighlyNullDataCheck")]

    # test np.array
    assert highly_null_check.validate(np.array([[None, None, None, None, 0], [None, None, None, "hi", 5]])) == [DataCheckWarning("Column '0' is 80.0% or more null", "HighlyNullDataCheck"),
                                                                                                                DataCheckWarning("Column '1' is 80.0% or more null", "HighlyNullDataCheck"),
                                                                                                                DataCheckWarning("Column '2' is 80.0% or more null", "HighlyNullDataCheck")]
