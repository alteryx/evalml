import numpy as np
import pandas as pd

from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.messsage import DataCheckError


def test_empty_data_check(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(self, X, y=None, verbose=True):
            return [], []

    data_check = MockDataCheck()
    errors, warnings = data_check.validate(X, y=y)
    assert len(errors) == 0
    assert len(warnings) == 0


def test_data_check_with_parameters():
    X = pd.DataFrame(np.array([[1, 2, -1], [1, 2, 0]]))

    class MockDataCheckWithParams(DataCheck):
        def __init__(self, less_than):
            self.less_than = less_than

        def validate(self, X, y=None, verbose=True):
            warnings = []
            errors = []
            if (X < self.less_than).any().any():
                errors.append(DataCheckError("Value less than zero!"))
            return errors, warnings

    data_check = MockDataCheckWithParams(less_than=0)
    errors, warnings = data_check.validate(X, y=None)
    assert len(errors) == 1
    assert len(warnings) == 0
