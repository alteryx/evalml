import numpy as np
import pandas as pd

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_checks import DataChecks
from evalml.data_checks.messsage import DataCheckError, DataCheckWarning


def test_data_checks(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y, verbose=True):
            return [], []

    class MockDataCheckWarning(DataCheck):
        def validate(self, X, y, verbose=True):
            return [], [DataCheckWarning("warning one")]

    class MockDataCheckError(DataCheck):
        def validate(self, X, y, verbose=True):
            return [DataCheckError("error one")], []

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y, verbose=True):
            return [DataCheckError("error two")], [DataCheckWarning("warning two")]

    data_checks_list = [MockDataCheck(), MockDataCheckWarning(), MockDataCheckError(), MockDataCheckErrorAndWarning()]
    data_checks = DataChecks(data_checks=data_checks_list)
    errors, warnings = data_checks.validate(X, y, verbose=True)
    assert len(errors) == 2
    assert len(warnings) == 2

    expected_error_msgs = set(["error one", "error two"])
    expected_warning_msgs = set(["warning one", "warning two"])
    actual_error_msgs = set([str(error) for error in errors])
    actual_warning_msgs = set([str(warning) for warning in warnings])

    assert actual_error_msgs == expected_error_msgs
    assert actual_warning_msgs == expected_warning_msgs


def test_data_checks_with_parameters():
    X = pd.DataFrame(np.array([[1, 2, -1], [1, 2, 0]]))

    class MockDataCheckLessThan(DataCheck):
        def __init__(self, less_than):
            self.less_than = less_than

        def validate(self, X, y=None, verbose=True):
            warnings = []
            errors = []
            if (X < self.less_than).any().any():
                errors.append(DataCheckError("There are values less than {}!".format(self.less_than)))
            return errors, warnings

    class MockDataCheckGreaterThan(DataCheck):
        def __init__(self, greater_than):
            self.greater_than = greater_than

        def validate(self, X, y=None, verbose=True):
            warnings = []
            errors = []
            if (X > self.greater_than).any().any():
                warnings.append(DataCheckWarning("There are values greater than {}!".format(self.greater_than)))
            return errors, warnings

    data_checks_list = [MockDataCheckLessThan(less_than=0), MockDataCheckGreaterThan(greater_than=1)]
    data_checks = DataChecks(data_checks=data_checks_list)
    errors, warnings = data_checks.validate(X, y=None, verbose=True)
    assert len(errors) == 1
    assert len(warnings) == 1
