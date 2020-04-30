import numpy as np
import pandas as pd

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)


def test_empty_data_check(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None, verbose=True):
            return [], []

    data_check = MockDataCheck()
    assert data_check.name == "MockDataCheck"

    errors, warnings = data_check.validate(X, y=y)
    assert len(errors) == 0
    assert len(warnings) == 0


def test_data_check_with_parameters():
    X = pd.DataFrame(np.array([[1, 2, -1], [1, 2, 0]]))

    class MockDataCheckWithParams(DataCheck):
        def __init__(self, less_than, greater_than):
            self.less_than = less_than
            self.greater_than = greater_than

        def validate(self, X, y=None, verbose=True):
            warnings = []
            errors = []
            if (X < self.less_than).any().any():
                errors.append(DataCheckError("There are values less than {}!".format(self.less_than), self.name))
            if (X > self.greater_than).any().any():
                warnings.append(DataCheckWarning("There are values greater than {}!".format(self.greater_than), self.name))
            return errors, warnings

    data_check = MockDataCheckWithParams(less_than=0, greater_than=1)
    assert data_check.name == "MockDataCheckWithParams"

    errors, warnings = data_check.validate(X, y=None)
    assert len(errors) == 1
    assert len(warnings) == 1
