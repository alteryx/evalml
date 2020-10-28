import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import DefaultDataChecks, EmptyDataChecks
from evalml.data_checks.data_check import DataCheck
from evalml.data_checks.data_check_message import (
    DataCheckError,
    DataCheckWarning
)
from evalml.data_checks.data_checks import AutoMLDataChecks, DataChecks
from evalml.exceptions import DataCheckInitError


def test_data_checks(X_y_binary):
    X, y = X_y_binary

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

    data_checks_list = [MockDataCheck, MockDataCheckWarning, MockDataCheckError, MockDataCheckErrorAndWarning]
    data_checks = DataChecks(data_checks=data_checks_list)
    assert data_checks.validate(X, y) == [DataCheckWarning("warning one", "MockDataCheckWarning"),
                                          DataCheckError("error one", "MockDataCheckError"),
                                          DataCheckError("error two", "MockDataCheckErrorAndWarning"),
                                          DataCheckWarning("warning two", "MockDataCheckErrorAndWarning")]


def test_empty_data_checks(X_y_binary):
    X, y = X_y_binary
    data_checks = EmptyDataChecks()
    assert data_checks.validate(X, y) == []


messages = [DataCheckWarning("Column 'all_null' is 95.0% or more null", "HighlyNullDataCheck"),
            DataCheckWarning("Column 'also_all_null' is 95.0% or more null", "HighlyNullDataCheck"),
            DataCheckWarning("Column 'id' is 100.0% or more likely to be an ID column", "IDColumnsDataCheck"),
            DataCheckError("1 row(s) (20.0%) of target values are null", "InvalidTargetDataCheck"),
            DataCheckError("lots_of_null has 1 unique value.", "NoVarianceDataCheck"),
            DataCheckError("all_null has 0 unique value.", "NoVarianceDataCheck"),
            DataCheckError("also_all_null has 0 unique value.", "NoVarianceDataCheck")]


def test_default_data_checks_classification():
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, "some data"],
                      'all_null': [None, None, None, None, None],
                      'also_all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5],
                      'id': [0, 1, 2, 3, 4],
                      'has_label_leakage': [100, 200, 100, 200, 100]})
    y = pd.Series([0, 1, np.nan, 1, 0])
    data_checks = DefaultDataChecks("binary")

    leakage = [DataCheckWarning("Column 'has_label_leakage' is 95.0% or more correlated with the target", "TargetLeakageDataCheck")]
    imbalance = [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [1.0, 0.0]", "ClassImbalanceDataCheck")]

    assert data_checks.validate(X, y) == messages[:3] + leakage + messages[3:] + imbalance

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "binary"}})
    assert data_checks.validate(X, y) == messages[:3] + leakage + messages[3:]

    imbalance = [DataCheckError("The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0.0, 2.0, 1.0]", "ClassImbalanceDataCheck")]
    # multiclass
    y = pd.Series([0, 1, np.nan, 2, 0])
    data_checks = DefaultDataChecks("multiclass")
    assert data_checks.validate(X, y) == messages + imbalance

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "multiclass"}})
    assert data_checks.validate(X, y) == messages


def test_default_data_checks_regression():
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, "some data"],
                      'all_null': [None, None, None, None, None],
                      'also_all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5],
                      'id': [0, 1, 2, 3, 4],
                      'has_label_leakage': [100, 200, 100, 200, 100]})
    y = pd.Series([0.3, 100.0, np.nan, 1.0, 0.2])
    y2 = pd.Series([5] * 4)

    data_checks = DefaultDataChecks("regression")
    assert data_checks.validate(X, y) == messages

    # Skip Invalid Target
    assert data_checks.validate(X, y2) == messages[:3] + messages[4:] + [DataCheckError("Y has 1 unique value.", "NoVarianceDataCheck")]

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "regression"}})
    assert data_checks.validate(X, y) == messages


def test_data_checks_init_from_classes():
    def make_mock_data_check(check_name):
        class MockCheck(DataCheck):
            name = check_name

            def __init__(self, foo, bar, baz=3):
                self.foo = foo
                self.bar = bar
                self.baz = baz

            def validate(self, X, y=None):
                """Mock validate."""

        return MockCheck
    data_checks = [make_mock_data_check("check_1"), make_mock_data_check("check_2")]
    checks = DataChecks(data_checks,
                        data_check_params={"check_1": {"foo": 1, "bar": 2},
                                           "check_2": {"foo": 3, "bar": 1, "baz": 4}})
    assert checks.data_checks[0].foo == 1
    assert checks.data_checks[0].bar == 2
    assert checks.data_checks[0].baz == 3
    assert checks.data_checks[1].foo == 3
    assert checks.data_checks[1].bar == 1
    assert checks.data_checks[1].baz == 4


class MockCheck(DataCheck):
    name = "mock_check"

    def __init__(self, foo, bar, baz=3):
        """Mock init"""

    def validate(self, X, y=None):
        """Mock validate."""


class MockCheck2(DataCheck):
    name = "MockCheck"

    def __init__(self, foo, bar, baz=3):
        """Mock init"""

    def validate(self, X, y=None):
        """Mock validate."""


@pytest.mark.parametrize("classes,params,expected_exception,expected_message",
                         [([MockCheck], {"mock_check": 1}, DataCheckInitError,
                           "Parameters for mock_check were not in a dictionary. Received 1."),
                          ([MockCheck], {"mock_check": {"foo": 1}}, DataCheckInitError,
                           r"Encountered the following error while initializing mock_check: __init__\(\) missing 1 required positional argument: 'bar'"),
                          ([MockCheck], {"mock_check": {"Bar": 2}}, DataCheckInitError,
                           r"Encountered the following error while initializing mock_check: __init__\(\) got an unexpected keyword argument 'Bar'"),
                          ([MockCheck], {"mock_check": {"fo": 3, "ba": 4}}, DataCheckInitError,
                           r"Encountered the following error while initializing mock_check: __init__\(\) got an unexpected keyword argument 'fo'"),
                          ([MockCheck], {"MockCheck": {"foo": 2, "bar": 4}}, DataCheckInitError,
                           "Class MockCheck was provided in params dictionary but it does not match any name in the data_check_classes list."),
                          ([MockCheck, MockCheck2], {"MockCheck": {"foo": 2, "bar": 4}}, DataCheckInitError,
                           "Class mock_check was provided in the data_checks_classes list but it does not have an entry in the parameters dictionary."),
                          ([1], None, ValueError, ("All elements of parameter data_checks must be an instance of DataCheck " +
                                                   "or a DataCheck class with any desired parameters specified in the " +
                                                   "data_check_params dictionary.")),
                          ([MockCheck], [1], ValueError, r"Params must be a dictionary. Received \[1\]")])
def test_data_checks_raises_value_errors_on_init(classes, params, expected_exception, expected_message):

    with pytest.raises(expected_exception, match=expected_message):
        DataChecks(classes, params)


def test_automl_data_checks_raises_value_error():
    with pytest.raises(ValueError, match="All elements of parameter data_checks must be an instance of DataCheck."):
        AutoMLDataChecks([1, MockCheck])
