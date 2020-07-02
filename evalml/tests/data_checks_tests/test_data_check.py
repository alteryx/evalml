import pandas as pd
import pytest

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
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


def test_data_check_validate_simple(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return [DataCheckError("error one", self.name), DataCheckWarning("warning one", self.name)]

    data_check = MockDataCheck()
    assert data_check.validate(X, y=y) == [DataCheckError("error one", "MockDataCheck"), DataCheckWarning("warning one", "MockDataCheck")]


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
    assert data_check.validate(X, y=None) == []

    data_check = MockDataCheckWithParam(num=0)
    assert data_check.validate(X, y=None) == [DataCheckError("Expected num == 10", "MockDataCheckWithParam")]
