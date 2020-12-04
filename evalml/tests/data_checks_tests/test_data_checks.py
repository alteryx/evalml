import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    AutoMLDataChecks,
    DataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataChecks,
    DataCheckWarning,
    DefaultDataChecks,
    EmptyDataChecks
)
from evalml.exceptions import DataCheckInitError


def test_data_checks(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return {"warnings": [], "errors": []}

    class MockDataCheckWarning(DataCheck):
        def validate(self, X, y):
            return {"warnings": [DataCheckWarning(message="warning one", data_check_name=self.name, message_code=None).to_dict()], "errors": []}

    class MockDataCheckError(DataCheck):
        def validate(self, X, y):
            return {"warnings": [], "errors": [DataCheckError(message="error one", data_check_name=self.name, message_code=None).to_dict()]}

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y):
            return {"warnings": [DataCheckWarning(message="warning two", data_check_name=self.name, message_code=None).to_dict()],
                    "errors": [DataCheckError(message="error two", data_check_name=self.name, message_code=None).to_dict()]}

    data_checks_list = [MockDataCheck, MockDataCheckWarning, MockDataCheckError, MockDataCheckErrorAndWarning]
    data_checks = DataChecks(data_checks=data_checks_list)
    assert data_checks.validate(X, y) == {
        "warnings": [DataCheckWarning(message="warning one", data_check_name="MockDataCheckWarning").to_dict(),
                     DataCheckWarning(message="warning two", data_check_name="MockDataCheckErrorAndWarning").to_dict()],
        "errors": [DataCheckError(message="error one", data_check_name="MockDataCheckError").to_dict(),
                   DataCheckError(message="error two", data_check_name="MockDataCheckErrorAndWarning").to_dict()]
    }


@pytest.mark.parametrize("input_type", ["pd", "ww", "np"])
def test_empty_data_checks(input_type, X_y_binary):
    X, y = X_y_binary
    if input_type != "np":
        X = pd.DataFrame(X)
        y = pd.Series(y)
    if input_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
    data_checks = EmptyDataChecks()
    assert data_checks.validate(X, y) == {"warnings": [], "errors": []}


messages = [DataCheckWarning(message="Column 'all_null' is 95.0% or more null",
                             data_check_name="HighlyNullDataCheck",
                             message_code=DataCheckMessageCode.HIGHLY_NULL,
                             details={"column": "all_null"}).to_dict(),
            DataCheckWarning(message="Column 'also_all_null' is 95.0% or more null",
                             data_check_name="HighlyNullDataCheck",
                             message_code=DataCheckMessageCode.HIGHLY_NULL,
                             details={"column": "also_all_null"}).to_dict(),
            DataCheckWarning(message="Column 'id' is 100.0% or more likely to be an ID column",
                             data_check_name="IDColumnsDataCheck",
                             message_code=DataCheckMessageCode.HAS_ID_COLUMN,
                             details={"column": "id"}).to_dict(),
            DataCheckError(message="1 row(s) (20.0%) of target values are null",
                           data_check_name="InvalidTargetDataCheck",
                           message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                           details={"num_null_rows": 1, "pct_null_rows": 20}).to_dict(),
            DataCheckError(message="lots_of_null has 1 unique value.",
                           data_check_name="NoVarianceDataCheck",
                           message_code=DataCheckMessageCode.NO_VARIANCE,
                           details={"column": "lots_of_null"}).to_dict(),
            DataCheckError(message="all_null has 0 unique value.",
                           data_check_name="NoVarianceDataCheck",
                           message_code=DataCheckMessageCode.NO_VARIANCE,
                           details={"column": "all_null"}).to_dict(),
            DataCheckError(message="also_all_null has 0 unique value.",
                           data_check_name="NoVarianceDataCheck",
                           message_code=DataCheckMessageCode.NO_VARIANCE,
                           details={"column": "also_all_null"}).to_dict()]


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_classification(input_type):
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, "some data"],
                      'all_null': [None, None, None, None, None],
                      'also_all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5],
                      'id': [0, 1, 2, 3, 4],
                      'has_label_leakage': [100, 200, 100, 200, 100]})
    y = pd.Series([0, 1, np.nan, 1, 0])
    y_multiclass = pd.Series([0, 1, np.nan, 2, 0])
    if input_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
        y_multiclass = ww.DataColumn(y_multiclass)

    data_checks = DefaultDataChecks("binary")

    leakage = [DataCheckWarning(message="Column 'has_label_leakage' is 95.0% or more correlated with the target",
                                data_check_name="TargetLeakageDataCheck",
                                message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                details={"column": "has_label_leakage"}).to_dict()]
    imbalance = [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [1.0, 0.0]",
                                data_check_name="ClassImbalanceDataCheck",
                                message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                details={"target_values": [1.0, 0.0]}).to_dict()]

    assert data_checks.validate(X, y) == {"warnings": messages[:3] + leakage, "errors": messages[3:] + imbalance}

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "binary"}})
    assert data_checks.validate(X, y) == {"warnings": messages[:3] + leakage, "errors": messages[3:]}

    imbalance = [DataCheckError(message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0.0, 2.0, 1.0]",
                                data_check_name="ClassImbalanceDataCheck",
                                message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
                                details={"target_values": [0.0, 2.0, 1.0]}).to_dict()]
    # multiclass
    data_checks = DefaultDataChecks("multiclass")
    assert data_checks.validate(X, y_multiclass) == {"warnings": messages[:3], "errors": messages[3:] + imbalance}

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "multiclass"}})
    assert data_checks.validate(X, y_multiclass) == {"warnings": messages[:3], "errors": messages[3:]}


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_regression(input_type):
    X = pd.DataFrame({'lots_of_null': [None, None, None, None, "some data"],
                      'all_null': [None, None, None, None, None],
                      'also_all_null': [None, None, None, None, None],
                      'no_null': [1, 2, 3, 4, 5],
                      'id': [0, 1, 2, 3, 4],
                      'has_label_leakage': [100, 200, 100, 200, 100]})
    y = pd.Series([0.3, 100.0, np.nan, 1.0, 0.2])
    y_no_variance = pd.Series([5] * 4)

    if input_type == "ww":
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
        y_no_variance = ww.DataColumn(y_no_variance)
    data_checks = DefaultDataChecks("regression")
    assert data_checks.validate(X, y) == {"warnings": messages[:3], "errors": messages[3:]}

    # Skip Invalid Target
    assert data_checks.validate(X, y_no_variance) == {"warnings": messages[:3], "errors": messages[4:] + [DataCheckError(message="Y has 1 unique value.",
                                                                                                                         data_check_name="NoVarianceDataCheck",
                                                                                                                         message_code=DataCheckMessageCode.NO_VARIANCE,
                                                                                                                         details={"column": "Y"}).to_dict()]}

    data_checks = DataChecks(DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
                             {"InvalidTargetDataCheck": {"problem_type": "regression"}})
    assert data_checks.validate(X, y) == {"warnings": messages[:3], "errors": messages[3:]}


def test_default_data_checks_time_series_regression():
    regression_data_check_classes = [check.__class__ for check in DefaultDataChecks("regression").data_checks]
    ts_regression_data_check_classes = [check.__class__ for check in DefaultDataChecks("time series regression").data_checks]
    assert regression_data_check_classes == ts_regression_data_check_classes


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
