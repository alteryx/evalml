from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_checks import DataChecks


def test_data_check(X_y):
    X, y = X_y

    class MockDataCheck(DataCheck):
        def validate(X, y, verbose=True):
            return [], []

    class MockDataCheckWarning(DataCheck):
        def validate(X, y, verbose=True):
            return [], []

    class MockDataCheckError(DataCheck):
        def validate(X, y, verbose=True):
            return [], []

    data_checks_list = [MockDataCheck(), MockDataCheckWarning(), MockDataCheckError()]
    data_checks = DataChecks(data_checks=data_checks_list)
    data_checks.validate(X, y, verbose=True)
