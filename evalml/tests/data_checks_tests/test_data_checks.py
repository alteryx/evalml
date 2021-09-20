from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.automl import get_default_primary_search_objective
from evalml.data_checks import (
    ClassImbalanceDataCheck,
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataChecks,
    DataCheckWarning,
    DateTimeFormatDataCheck,
    DefaultDataChecks,
    TargetDistributionDataCheck,
)
from evalml.exceptions import DataCheckInitError
from evalml.problem_types import ProblemTypes, is_time_series


def test_data_checks_not_list_error(X_y_binary):
    with pytest.raises(ValueError, match="Parameter data_checks must be a list."):
        DataChecks(data_checks=1)


def test_data_checks(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return {"warnings": [], "errors": [], "actions": []}

    class MockDataCheckWarning(DataCheck):
        def validate(self, X, y):
            return {
                "warnings": [
                    DataCheckWarning(
                        message="warning one",
                        data_check_name=self.name,
                        message_code=None,
                    ).to_dict()
                ],
                "errors": [],
                "actions": [],
            }

    class MockDataCheckError(DataCheck):
        def validate(self, X, y):
            return {
                "warnings": [],
                "errors": [
                    DataCheckError(
                        message="error one",
                        data_check_name=self.name,
                        message_code=None,
                    ).to_dict()
                ],
                "actions": [],
            }

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y):
            return {
                "warnings": [
                    DataCheckWarning(
                        message="warning two",
                        data_check_name=self.name,
                        message_code=None,
                    ).to_dict()
                ],
                "errors": [
                    DataCheckError(
                        message="error two",
                        data_check_name=self.name,
                        message_code=None,
                    ).to_dict()
                ],
                "actions": [],
            }

    data_checks_list = [
        MockDataCheck,
        MockDataCheckWarning,
        MockDataCheckError,
        MockDataCheckErrorAndWarning,
    ]
    data_checks = DataChecks(data_checks=data_checks_list)
    assert data_checks.validate(X, y) == {
        "warnings": [
            DataCheckWarning(
                message="warning one", data_check_name="MockDataCheckWarning"
            ).to_dict(),
            DataCheckWarning(
                message="warning two", data_check_name="MockDataCheckErrorAndWarning"
            ).to_dict(),
        ],
        "errors": [
            DataCheckError(
                message="error one", data_check_name="MockDataCheckError"
            ).to_dict(),
            DataCheckError(
                message="error two", data_check_name="MockDataCheckErrorAndWarning"
            ).to_dict(),
        ],
        "actions": [],
    }


messages = [
    DataCheckWarning(
        message="Column 'all_null' is 95.0% or more null",
        data_check_name="HighlyNullDataCheck",
        message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
        details={"column": "all_null", "pct_null_rows": 1.0},
    ).to_dict(),
    DataCheckWarning(
        message="Column 'also_all_null' is 95.0% or more null",
        data_check_name="HighlyNullDataCheck",
        message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
        details={"column": "also_all_null", "pct_null_rows": 1.0},
    ).to_dict(),
    DataCheckWarning(
        message="Column 'id' is 100.0% or more likely to be an ID column",
        data_check_name="IDColumnsDataCheck",
        message_code=DataCheckMessageCode.HAS_ID_COLUMN,
        details={"column": "id"},
    ).to_dict(),
    DataCheckError(
        message="1 row(s) (20.0%) of target values are null",
        data_check_name="InvalidTargetDataCheck",
        message_code=DataCheckMessageCode.TARGET_HAS_NULL,
        details={"num_null_rows": 1, "pct_null_rows": 20.0},
    ).to_dict(),
    DataCheckError(
        message="lots_of_null has 1 unique value.",
        data_check_name="NoVarianceDataCheck",
        message_code=DataCheckMessageCode.NO_VARIANCE,
        details={"column": "lots_of_null"},
    ).to_dict(),
    DataCheckError(
        message="all_null has 0 unique value.",
        data_check_name="NoVarianceDataCheck",
        message_code=DataCheckMessageCode.NO_VARIANCE,
        details={"column": "all_null"},
    ).to_dict(),
    DataCheckError(
        message="also_all_null has 0 unique value.",
        data_check_name="NoVarianceDataCheck",
        message_code=DataCheckMessageCode.NO_VARIANCE,
        details={"column": "also_all_null"},
    ).to_dict(),
    DataCheckError(
        message="Input natural language column(s) (natural_language_nan) contains NaN values. Please impute NaN values or drop these rows or columns.",
        data_check_name="NaturalLanguageNaNDataCheck",
        message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
        details={"columns": "natural_language_nan"},
    ).to_dict(),
    DataCheckError(
        message="Input datetime column(s) (nan_dt_col) contains NaN values. Please impute NaN values or drop these rows or columns.",
        data_check_name="DateTimeNaNDataCheck",
        message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
        details={"columns": "nan_dt_col"},
    ).to_dict(),
]

expected_actions = [
    DataCheckAction(
        DataCheckActionCode.DROP_COL, metadata={"column": "all_null"}
    ).to_dict(),
    DataCheckAction(
        DataCheckActionCode.DROP_COL, metadata={"column": "also_all_null"}
    ).to_dict(),
    DataCheckAction(DataCheckActionCode.DROP_COL, metadata={"column": "id"}).to_dict(),
    DataCheckAction(
        DataCheckActionCode.IMPUTE_COL,
        metadata={
            "column": None,
            "is_target": True,
            "impute_strategy": "most_frequent",
        },
    ).to_dict(),
    DataCheckAction(
        DataCheckActionCode.DROP_COL, metadata={"column": "lots_of_null"}
    ).to_dict(),
]


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_classification(input_type):
    X = pd.DataFrame(
        {
            "lots_of_null": [None, None, None, None, "some data"],
            "all_null": [None, None, None, None, None],
            "also_all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
            "id": [0, 1, 2, 3, 4],
            "has_label_leakage": [100, 200, 100, 200, 100],
            "natural_language_nan": [
                None,
                "string_that_is_long_enough_for_natural_language_1",
                "string_that_is_long_enough_for_natural_language_2",
                "string_that_is_long_enough_for_natural_language_3",
                "string_that_is_long_enough_for_natural_language_4",
            ],
            "nan_dt_col": pd.Series(pd.date_range("20200101", periods=5)),
        }
    )
    X["nan_dt_col"][0] = None

    y = pd.Series([0, 1, np.nan, 1, 0])
    y_multiclass = pd.Series([0, 1, np.nan, 2, 0])
    X.ww.init(logical_types={"natural_language_nan": "NaturalLanguage"})
    if input_type == "ww":
        y = ww.init_series(y)
        y_multiclass = ww.init_series(y_multiclass)

    data_checks = DefaultDataChecks(
        "binary", get_default_primary_search_objective("binary")
    )
    imbalance = [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0.0, 1.0]",
            data_check_name="ClassImbalanceDataCheck",
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0.0, 1.0]},
        ).to_dict()
    ]

    assert data_checks.validate(X, y) == {
        "warnings": messages[:3],
        "errors": messages[3:] + imbalance,
        "actions": expected_actions,
    }

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "binary",
                "objective": get_default_primary_search_objective("binary"),
            }
        },
    )
    assert data_checks.validate(X, y) == {
        "warnings": messages[:3],
        "errors": messages[3:],
        "actions": expected_actions,
    }

    # multiclass
    imbalance = [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0.0, 1.0, 2.0]",
            data_check_name="ClassImbalanceDataCheck",
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0.0, 1.0, 2.0]},
        ).to_dict()
    ]
    min_2_class_count = [
        DataCheckError(
            message="Target does not have at least two instances per class which is required for multiclass classification",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS,
            details={"least_populated_class_labels": [1.0, 2.0]},
        ).to_dict()
    ]
    high_class_to_sample_ratio = [
        DataCheckWarning(
            message="Target has a large number of unique values, could be regression type problem.",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
            details={"class_to_value_ratio": 0.6},
        ).to_dict()
    ]
    # multiclass
    data_checks = DefaultDataChecks(
        "multiclass", get_default_primary_search_objective("multiclass")
    )
    assert data_checks.validate(X, y_multiclass) == {
        "warnings": messages[:3] + high_class_to_sample_ratio,
        "errors": [messages[3]] + min_2_class_count + messages[4:] + imbalance,
        "actions": expected_actions,
    }

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "multiclass",
                "objective": get_default_primary_search_objective("multiclass"),
            }
        },
    )
    assert data_checks.validate(X, y_multiclass) == {
        "warnings": messages[:3] + high_class_to_sample_ratio,
        "errors": [messages[3]] + min_2_class_count + messages[4:],
        "actions": expected_actions,
    }


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_regression(input_type):
    X = pd.DataFrame(
        {
            "lots_of_null": [None, None, None, None, "some data"],
            "all_null": [None, None, None, None, None],
            "also_all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 5, 5],
            "id": [0, 1, 2, 3, 4],
            "has_label_leakage": [100, 200, 100, 200, 100],
            "natural_language_nan": [
                None,
                "string_that_is_long_enough_for_natural_language_1",
                "string_that_is_long_enough_for_natural_language_2",
                "string_that_is_long_enough_for_natural_language_3",
                "string_that_is_long_enough_for_natural_language_4",
            ],
            "nan_dt_col": pd.Series(pd.date_range("20200101", periods=5)),
        }
    )
    X["nan_dt_col"][0] = None
    y = pd.Series([0.3, 100.0, np.nan, 1.0, 0.2])
    y_no_variance = pd.Series([5] * 5)
    X.ww.init(
        logical_types={
            "lots_of_null": "categorical",
            "natural_language_nan": "NaturalLanguage",
        }
    )
    if input_type == "ww":
        y = ww.init_series(y)
        y_no_variance = ww.init_series(y_no_variance)
    null_leakage = [
        DataCheckWarning(
            message="Column 'lots_of_null' is 95.0% or more correlated with the target",
            data_check_name="TargetLeakageDataCheck",
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"column": "lots_of_null"},
        ).to_dict()
    ]
    data_checks = DefaultDataChecks(
        "regression", get_default_primary_search_objective("regression")
    )
    id_leakage_warning = [
        DataCheckWarning(
            message="Column 'id' is 95.0% or more correlated with the target",
            data_check_name="TargetLeakageDataCheck",
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"column": "id"},
        ).to_dict()
    ]
    nan_dt_leakage_warning = [
        DataCheckWarning(
            message="Column 'nan_dt_col' is 95.0% or more correlated with the target",
            data_check_name="TargetLeakageDataCheck",
            message_code=DataCheckMessageCode.TARGET_LEAKAGE,
            details={"column": "nan_dt_col"},
        ).to_dict()
    ]

    impute_action = DataCheckAction(
        DataCheckActionCode.IMPUTE_COL,
        metadata={"column": None, "is_target": True, "impute_strategy": "mean"},
    ).to_dict()
    nan_dt_action = DataCheckAction(
        DataCheckActionCode.DROP_COL, metadata={"column": "nan_dt_col"}
    ).to_dict()
    expected_actions_with_drop_and_impute = (
        expected_actions[:3] + [nan_dt_action, impute_action] + expected_actions[4:]
    )
    assert data_checks.validate(X, y) == {
        "warnings": messages[:3] + id_leakage_warning + nan_dt_leakage_warning,
        "errors": messages[3:],
        "actions": expected_actions_with_drop_and_impute,
    }

    # Skip Invalid Target
    assert data_checks.validate(X, y_no_variance) == {
        "warnings": messages[:3] + null_leakage,
        "errors": messages[4:7]
        + [
            DataCheckError(
                message="Y has 1 unique value.",
                data_check_name="NoVarianceDataCheck",
                message_code=DataCheckMessageCode.NO_VARIANCE,
                details={"column": "Y"},
            ).to_dict()
        ]
        + messages[7:],
        "actions": expected_actions[:3] + expected_actions[4:],
    }

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "regression",
                "objective": get_default_primary_search_objective("regression"),
            }
        },
    )
    assert data_checks.validate(X, y) == {
        "warnings": messages[:3] + id_leakage_warning + nan_dt_leakage_warning,
        "errors": messages[3:],
        "actions": expected_actions_with_drop_and_impute,
    }


def test_default_data_checks_null_rows():
    class SeriesWrap:
        def __init__(self, series):
            self.series = series

        def __eq__(self, series_2):
            return all(self.series.eq(series_2.series))

    X = pd.DataFrame(
        {
            "all_null": [None, None, None, None, None],
            "also_all_null": [None, None, None, None, None],
        }
    )
    y = pd.Series([0, 1, np.nan, 1, 0])
    data_checks = DefaultDataChecks(
        "regression", get_default_primary_search_objective("regression")
    )
    highly_null_rows = SeriesWrap(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0]))
    expected = {
        "warnings": [
            DataCheckWarning(
                message="5 out of 5 rows are more than 95.0% null",
                data_check_name="HighlyNullDataCheck",
                message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                details={"pct_null_cols": highly_null_rows},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'all_null' is 95.0% or more null",
                data_check_name="HighlyNullDataCheck",
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={"column": "all_null", "pct_null_rows": 1.0},
            ).to_dict(),
            DataCheckWarning(
                message="Column 'also_all_null' is 95.0% or more null",
                data_check_name="HighlyNullDataCheck",
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={"column": "also_all_null", "pct_null_rows": 1.0},
            ).to_dict(),
        ],
        "errors": [
            DataCheckError(
                message="1 row(s) (20.0%) of target values are null",
                data_check_name="InvalidTargetDataCheck",
                message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                details={"num_null_rows": 1, "pct_null_rows": 20.0},
            ).to_dict(),
            DataCheckError(
                message="all_null has 0 unique value.",
                data_check_name="NoVarianceDataCheck",
                message_code=DataCheckMessageCode.NO_VARIANCE,
                details={"column": "all_null"},
            ).to_dict(),
            DataCheckError(
                message="also_all_null has 0 unique value.",
                data_check_name="NoVarianceDataCheck",
                message_code=DataCheckMessageCode.NO_VARIANCE,
                details={"column": "also_all_null"},
            ).to_dict(),
        ],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_ROWS, metadata={"rows": [0, 1, 2, 3, 4]}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "all_null"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "also_all_null"}
            ).to_dict(),
            DataCheckAction(
                DataCheckActionCode.IMPUTE_COL,
                metadata={"column": None, "is_target": True, "impute_strategy": "mean"},
            ).to_dict(),
        ],
    }
    validation_results = data_checks.validate(X, y)
    validation_results["warnings"][0]["details"]["pct_null_cols"] = SeriesWrap(
        validation_results["warnings"][0]["details"]["pct_null_cols"]
    )
    assert validation_results == expected


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_default_data_checks_across_problem_types(problem_type):
    default_data_check_list = DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES

    if is_time_series(problem_type):
        default_data_check_list = default_data_check_list + [DateTimeFormatDataCheck]
    if problem_type in [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]:
        default_data_check_list = default_data_check_list + [
            TargetDistributionDataCheck
        ]
    else:
        default_data_check_list = default_data_check_list + [ClassImbalanceDataCheck]

    data_check_classes = [
        check.__class__
        for check in DefaultDataChecks(
            problem_type, get_default_primary_search_objective(problem_type)
        ).data_checks
    ]

    assert data_check_classes == default_data_check_list


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
    checks = DataChecks(
        data_checks,
        data_check_params={
            "check_1": {"foo": 1, "bar": 2},
            "check_2": {"foo": 3, "bar": 1, "baz": 4},
        },
    )
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


@pytest.mark.parametrize(
    "classes,params,expected_exception,expected_message",
    [
        (
            [MockCheck],
            {"mock_check": 1},
            DataCheckInitError,
            "Parameters for mock_check were not in a dictionary. Received 1.",
        ),
        (
            [MockCheck],
            {"mock_check": {"foo": 1}},
            DataCheckInitError,
            r"Encountered the following error while initializing mock_check: __init__\(\) missing 1 required positional argument: 'bar'",
        ),
        (
            [MockCheck],
            {"mock_check": {"Bar": 2}},
            DataCheckInitError,
            r"Encountered the following error while initializing mock_check: __init__\(\) got an unexpected keyword argument 'Bar'",
        ),
        (
            [MockCheck],
            {"mock_check": {"fo": 3, "ba": 4}},
            DataCheckInitError,
            r"Encountered the following error while initializing mock_check: __init__\(\) got an unexpected keyword argument 'fo'",
        ),
        (
            [MockCheck],
            {"MockCheck": {"foo": 2, "bar": 4}},
            DataCheckInitError,
            "Class MockCheck was provided in params dictionary but it does not match any name in the data_check_classes list.",
        ),
        (
            [MockCheck, MockCheck2],
            {"MockCheck": {"foo": 2, "bar": 4}},
            DataCheckInitError,
            "Class mock_check was provided in the data_checks_classes list but it does not have an entry in the parameters dictionary.",
        ),
        (
            [1],
            None,
            ValueError,
            (
                "All elements of parameter data_checks must be an instance of DataCheck "
                + "or a DataCheck class with any desired parameters specified in the "
                + "data_check_params dictionary."
            ),
        ),
        ([MockCheck], [1], ValueError, r"Params must be a dictionary. Received \[1\]"),
    ],
)
def test_data_checks_raises_value_errors_on_init(
    classes, params, expected_exception, expected_message
):
    with pytest.raises(expected_exception, match=expected_message):
        DataChecks(classes, params)


@pytest.mark.parametrize(
    "objective",
    [
        "Root Mean Squared Log Error",
        "Mean Squared Log Error",
        "Mean Absolute Percentage Error",
    ],
)
def test_errors_warnings_in_invalid_target_data_check(objective, ts_data):
    X, y = ts_data
    y[0] = -1
    y = pd.Series(y)
    details = {"Count of offending values": sum(val <= 0 for val in y.values.flatten())}
    data_check_error = DataCheckError(
        message=f"Target has non-positive values which is not supported for {objective}",
        data_check_name="InvalidTargetDataCheck",
        message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
        details=details,
    ).to_dict()

    default_data_check = DefaultDataChecks(
        problem_type="time series regression", objective=objective
    ).data_checks
    for check in default_data_check:
        if check.name == "InvalidTargetDataCheck":
            assert check.validate(X, y) == {
                "warnings": [],
                "errors": [data_check_error],
                "actions": [],
            }


def test_data_checks_do_not_duplicate_actions(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return {
                "warnings": [],
                "errors": [],
                "actions": [
                    DataCheckAction(
                        DataCheckActionCode.DROP_COL, metadata={"column": "col_to_drop"}
                    ).to_dict()
                ],
            }

    class MockDataCheckWithSameAction(DataCheck):
        def validate(self, X, y):
            return {"warnings": [], "errors": [], "actions": []}

    data_checks_list = [MockDataCheck, MockDataCheckWithSameAction]
    data_checks = DataChecks(data_checks=data_checks_list)

    # Check duplicate actions are returned once
    assert data_checks.validate(X, y) == {
        "warnings": [],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"column": "col_to_drop"}
            ).to_dict()
        ],
    }


def test_data_checks_drop_index(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X["index_col"] = pd.Series(range(len(X)))
    X.ww.init(index="index_col")

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return {"warnings": [], "errors": [], "actions": []}

    assert MockDataCheck().validate(X, y)

    MockDataCheck.validate = MagicMock()
    checks = DataChecks([MockDataCheck, MockDataCheck, MockDataCheck])
    checks.validate(X, y)

    validate_args = MockDataCheck.validate.call_args_list
    for arg in validate_args:
        assert "index_col" not in arg[0][0].columns
