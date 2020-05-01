import numpy as np
import pandas as pd

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)
from evalml.data_checks.data_checks import DataChecks


def test_data_checks(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return []

    class MockDataCheckWarning(DataCheck):
        def validate(self, X, y):
            return [DataCheckWarning("warning one", self.name)]

    class MockDataCheckError(DataCheck):
        def validate(self, X, y):
            return [DataCheckError("error one", self.name)]

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y):
            return [DataCheckError("error two", self.name), DataCheckWarning("warning two", self.name)]

    data_checks_list = [MockDataCheck(), MockDataCheckWarning(), MockDataCheckError(), MockDataCheckErrorAndWarning()]
    data_checks = DataChecks(data_checks=data_checks_list)
    errors_warnings = data_checks.validate(X, y)
    assert len(errors_warnings) == 4


def test_data_checks_with_parameters():
    X = pd.DataFrame(np.array([[1, 2, -1], [1, 2, 0]]))

    class MockDataCheckLessThan(DataCheck):
        def __init__(self, less_than):
            self.less_than = less_than

        def validate(self, X, y=None):
            errors_warnings = []
            if (X < self.less_than).any().any():
                errors_warnings.append(DataCheckError("There are values less than {}!".format(self.less_than), self.name))
            return errors_warnings

    class MockDataCheckGreaterThan(DataCheck):
        def __init__(self, greater_than):
            self.greater_than = greater_than

        def validate(self, X, y=None):
            errors_warnings = []
            if (X > self.greater_than).any().any():
                errors_warnings.append(DataCheckWarning("There are values greater than {}!".format(self.greater_than), self.name))
            return errors_warnings

    data_checks_list = [MockDataCheckLessThan(less_than=0), MockDataCheckGreaterThan(greater_than=1)]
    data_checks = DataChecks(data_checks=data_checks_list)
    errors_warnings = data_checks.validate(X, y=None)
    assert len(errors_warnings) == 2
