import numpy as np
import pandas as pd

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)


def test_data_check_name():
    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return []
    assert MockDataCheck().name == "MockDataCheck"
    assert MockDataCheck.name == "MockDataCheck"
    errors_warnings = MockDataCheck().validate(pd.DataFrame())
    assert errors_warnings == []

    class Funky_Name1DataCheck(DataCheck):
        def validate(self, X, y=None):
            return []
    assert Funky_Name1DataCheck().name == "Funky_Name1DataCheck"
    assert Funky_Name1DataCheck.name == "Funky_Name1DataCheck"
    errors_warnings = Funky_Name1DataCheck().validate(pd.DataFrame())
    assert errors_warnings == []


def test_data_check_validate_empty(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None):
            return []

    data_check = MockDataCheck()
    errors_warnings = data_check.validate(X, y=y)
    assert errors_warnings == []


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
