import numpy as np
import pandas as pd
import pytest

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)
from evalml.data_checks.detect_highly_null_data_check import (
    DetectHighlyNullDataCheck
)


@pytest.fixture
def mock_data_check_class():
    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return []
    return MockDataCheck


def test_data_check_name(mock_data_check_class):
    assert mock_data_check_class().name == "MockDataCheck"
    assert mock_data_check_class.name == "MockDataCheck"

    class Funky_Name1DataCheck(mock_data_check_class):
        """Mock data check with a funky name"""

    assert Funky_Name1DataCheck().name == "Funky_Name1DataCheck"
    assert Funky_Name1DataCheck.name == "Funky_Name1DataCheck"


def test_empty_data_check_validate(mock_data_check_class):
    assert mock_data_check_class().validate(pd.DataFrame()) == []


def test_data_check_validate_simple(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return [DataCheckError("error one", self.name), DataCheckWarning("warning one", self.name)]

    data_check = MockDataCheck()
    errors_warnings = data_check.validate(X, y=y)
    assert errors_warnings == [DataCheckError("error one", "MockDataCheck"), DataCheckWarning("warning one", "MockDataCheck")]


def test_data_check_with_param():
    X = pd.DataFrame()

    class MockDataCheckWithParam(DataCheck):
        def __init__(self, num):
            self.num = num

        def validate(self, X, y=None):
            if self.num != 10:
                return [DataCheckError("Expected num == 10", self.name)]
            return []

    data_check = MockDataCheckWithParam(num=10)
    errors_warnings = data_check.validate(X, y=None)
    assert errors_warnings == []

    data_check = MockDataCheckWithParam(num=0)
    errors_warnings = data_check.validate(X, y=None)
    assert errors_warnings == [DataCheckError("Expected num == 10", "MockDataCheckWithParam")]


def test_highly_null_data_check_init():
    highly_null_check = DetectHighlyNullDataCheck()
    assert highly_null_check.percent_threshold == 0.95

    highly_null_check = DetectHighlyNullDataCheck(percent_threshold=0.0)
    assert highly_null_check.percent_threshold == 0

    highly_null_check = DetectHighlyNullDataCheck(percent_threshold=0.5)
    assert highly_null_check.percent_threshold == 0.5

    highly_null_check = DetectHighlyNullDataCheck(percent_threshold=1.0)
    assert highly_null_check.percent_threshold == 1.0

    with pytest.raises(ValueError, match="percent_threshold must be a float between 0 and 1, inclusive."):
        DetectHighlyNullDataCheck(percent_threshold=-0.1)
    with pytest.raises(ValueError, match="percent_threshold must be a float between 0 and 1, inclusive."):
        DetectHighlyNullDataCheck(percent_threshold=1.1)


def test_highly_null_data_check_warnings():
    data = pd.DataFrame({'lots_of_null': [None, None, None, None, 5],
                         'all_null': [None, None, None, None, None],
                         'no_null': [1, 2, 3, 4, 5]})
    no_null_check = DetectHighlyNullDataCheck(percent_threshold=0.0)
    assert no_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is 0.0% or more null", "DetectHighlyNullDataCheck"),
                                            DataCheckWarning("Column 'all_null' is 0.0% or more null", "DetectHighlyNullDataCheck"),
                                            DataCheckWarning("Column 'no_null' is 0.0% or more null", "DetectHighlyNullDataCheck")]
    some_null_check = DetectHighlyNullDataCheck(percent_threshold=0.5)
    assert some_null_check.validate(data) == [DataCheckWarning("Column 'lots_of_null' is 50.0% or more null", "DetectHighlyNullDataCheck"),
                                              DataCheckWarning("Column 'all_null' is 50.0% or more null", "DetectHighlyNullDataCheck")]
    all_null_check = DetectHighlyNullDataCheck(percent_threshold=1.0)
    assert all_null_check.validate(data) == [DataCheckWarning("Column 'all_null' is 100.0% or more null", "DetectHighlyNullDataCheck")]


def test_highly_null_data_check_input_formats():
    highly_null_check = DetectHighlyNullDataCheck(percent_threshold=0.8)

    # test empty pd.DataFrame
    messages = highly_null_check.validate(pd.DataFrame())
    assert messages == []

    #  test list
    messages = highly_null_check.validate([None, None, None, None, 5])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectHighlyNullDataCheck")]

    #  test pd.Series
    messages = highly_null_check.validate(pd.Series([None, None, None, None, 5]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectHighlyNullDataCheck")]

    #  test 2D list
    messages = highly_null_check.validate([[None, None, None, None, 0], [None, None, None, "hi", 5]])
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectHighlyNullDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectHighlyNullDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectHighlyNullDataCheck")]

    # test np.array
    messages = highly_null_check.validate(np.array([[None, None, None, None, 0], [None, None, None, "hi", 5]]))
    assert messages == [DataCheckWarning("Column '0' is 80.0% or more null", "DetectHighlyNullDataCheck"),
                        DataCheckWarning("Column '1' is 80.0% or more null", "DetectHighlyNullDataCheck"),
                        DataCheckWarning("Column '2' is 80.0% or more null", "DetectHighlyNullDataCheck")]
