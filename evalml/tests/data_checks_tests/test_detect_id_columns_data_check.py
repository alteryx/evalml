import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check_message import DataCheckWarning
from evalml.data_checks.detect_id_columns_data_check import (
    DetectIDColumnsDataCheck
)


def test_id_cols_data_check_init():
    id_cols_check = DetectIDColumnsDataCheck()
    assert id_cols_check.id_threshold == 1.0

    id_cols_check = DetectIDColumnsDataCheck(id_threshold=0.0)
    assert id_cols_check.id_threshold == 0

    id_cols_check = DetectIDColumnsDataCheck(id_threshold=0.5)
    assert id_cols_check.id_threshold == 0.5

    id_cols_check = DetectIDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.id_threshold == 1.0

    with pytest.raises(ValueError, match="id_threshold must be a float between 0 and 1, inclusive."):
        DetectIDColumnsDataCheck(id_threshold=-0.1)
    with pytest.raises(ValueError, match="id_threshold must be a float between 0 and 1, inclusive."):
        DetectIDColumnsDataCheck(id_threshold=1.1)


def test_detect_id_columns_warning():
    X_dict = {'col_1_id': [0, 1, 2, 3],
              'col_2': [2, 3, 4, 5],
              'col_3_id': [1, 1, 2, 3],
              'Id': [3, 1, 2, 0],
              'col_5': [0, 0, 1, 2],
              'col_6': [0.1, 0.2, 0.3, 0.4]
              }
    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = DetectIDColumnsDataCheck(id_threshold=0.95)

    assert id_cols_check.validate(X) == [DataCheckWarning("Column 'Id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_1_id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_2' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_3_id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]

    X = pd.DataFrame.from_dict(X_dict)
    id_cols_check = DetectIDColumnsDataCheck(id_threshold=1.0)

    assert id_cols_check.validate(X) == [DataCheckWarning("Column 'Id' is 100.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_1_id' is 100.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]


def test_detect_id_columns_strings():
    X_dict = {'col_1_id': ["a", "b", "c", "d"],
              'col_2': ["w", "x", "y", "z"],
              'col_3_id': ["a", "a", "b", "d"],
              'Id': ["z", "y", "x", "a"],
              'col_5': ["0", "0", "1", "2"],
              'col_6': [0.1, 0.2, 0.3, 0.4]
              }
    X = pd.DataFrame.from_dict(X_dict)

    id_cols_check = DetectIDColumnsDataCheck(id_threshold=0.95)

    assert id_cols_check.validate(X) == [DataCheckWarning("Column 'Id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_1_id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_2' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_3_id' is 95.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]

    id_cols_check = DetectIDColumnsDataCheck(id_threshold=1.0)
    assert id_cols_check.validate(X) == [DataCheckWarning("Column 'Id' is 100.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                                         DataCheckWarning("Column 'col_1_id' is 100.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]


def test_id_cols_data_check_input_formats():
    id_cols_check = DetectIDColumnsDataCheck(id_threshold=0.8)

    # test empty pd.DataFrame
    messages = id_cols_check.validate(pd.DataFrame())
    assert messages == []

    #  test list
    messages = id_cols_check.validate([1, 2, 3, 4, 5])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]

    #  test pd.Series
    messages = id_cols_check.validate(pd.Series([1, 2, 3, 4, 5]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]

    #  test 2D list
    messages = id_cols_check.validate([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]

    # test np.array
    messages = id_cols_check.validate(np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more likely to be an ID column", "DetectIDColumnsDataCheck")]
