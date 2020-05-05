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
    assert errors_warnings == [DataCheckWarning("warning one", "MockDataCheckWarning"),
                               DataCheckError("error one", "MockDataCheckError"),
                               DataCheckError("error two", "MockDataCheckErrorAndWarning"),
                               DataCheckWarning("warning two", "MockDataCheckErrorAndWarning")]
