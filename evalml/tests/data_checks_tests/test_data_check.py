import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataCheckResults,
    DataCheckWarning
)


@pytest.fixture
def mock_data_check_class():
    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return DataCheckResults()
    return MockDataCheck


def test_data_check_name(mock_data_check_class):
    assert mock_data_check_class().name == "MockDataCheck"
    assert mock_data_check_class.name == "MockDataCheck"

    class Funky_Name1DataCheck(mock_data_check_class):
        """Mock data check with a funky name"""

    assert Funky_Name1DataCheck().name == "Funky_Name1DataCheck"
    assert Funky_Name1DataCheck.name == "Funky_Name1DataCheck"


def test_empty_data_check_validate(mock_data_check_class):
    assert mock_data_check_class().validate(pd.DataFrame()) == DataCheckResults()


def test_data_check_validate_simple(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return DataCheckResults(errors=[DataCheckError("error one", self.name)],
                                    warnings=[DataCheckWarning("warning one", self.name)])

    data_check = MockDataCheck()
    assert data_check.validate(X, y=y) == DataCheckResults(errors=[DataCheckError("error one", "MockDataCheck")],
                                                           warnings=[DataCheckWarning("warning one", "MockDataCheck")])


def test_data_check_with_param():
    X = pd.DataFrame()

    class MockDataCheckWithParam(DataCheck):
        def __init__(self, num):
            self.num = num

        def validate(self, X, y=None):
            if self.num != 10:
                return DataCheckResults(errors=[DataCheckError("Expected num == 10", self.name)])
            return DataCheckResults()

    data_check = MockDataCheckWithParam(num=10)
    assert data_check.validate(X, y=None) == DataCheckResults()

    data_check = MockDataCheckWithParam(num=0)
    assert data_check.validate(X, y=None) == DataCheckResults(errors=[DataCheckError("Expected num == 10", "MockDataCheckWithParam")])
