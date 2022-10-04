from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.automl import get_default_primary_search_objective
from evalml.data_checks import (
    ClassImbalanceDataCheck,
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DataChecks,
    DataCheckWarning,
    DateTimeFormatDataCheck,
    DCAOParameterType,
    DefaultDataChecks,
    InvalidTargetDataCheck,
    TargetDistributionDataCheck,
    TimeSeriesParametersDataCheck,
    TimeSeriesSplittingDataCheck,
)
from evalml.exceptions import DataCheckInitError
from evalml.problem_types import (
    ProblemTypes,
    is_classification,
    is_regression,
    is_time_series,
)


@pytest.fixture
def data_checks_input_dataframe():
    X = pd.DataFrame(
        {
            "lots_of_null": [None, None, None, None, "some data"],
            "all_null": [None, None, None, None, None],
            "also_all_null": [None, None, None, None, None],
            "nullable_integer": [None, 2, 3, 4, 5],
            "nullable_bool": [None, True, False, True, True],
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
        },
    )
    X["nan_dt_col"][0] = None
    X.ww.init(
        logical_types={
            "lots_of_null": "categorical",
            "natural_language_nan": "NaturalLanguage",
            "nullable_integer": "IntegerNullable",
            "nullable_bool": "BooleanNullable",
        },
    )
    return X


def test_data_checks_not_list_error():
    with pytest.raises(ValueError, match="Parameter data_checks must be a list."):
        DataChecks(data_checks=1)


def test_data_checks(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return []

    class MockDataCheckWarning(DataCheck):
        def validate(self, X, y):
            return [
                DataCheckWarning(
                    message="warning one",
                    data_check_name=self.name,
                    message_code=None,
                ).to_dict(),
            ]

    class MockDataCheckError(DataCheck):
        def validate(self, X, y):
            return [
                DataCheckError(
                    message="error one",
                    data_check_name=self.name,
                    message_code=None,
                ).to_dict(),
            ]

    class MockDataCheckErrorAndWarning(DataCheck):
        def validate(self, X, y):
            return [
                DataCheckWarning(
                    message="warning two",
                    data_check_name=self.name,
                    message_code=None,
                ).to_dict(),
                DataCheckError(
                    message="error two",
                    data_check_name=self.name,
                    message_code=None,
                ).to_dict(),
            ]

    data_checks_list = [
        MockDataCheck,
        MockDataCheckWarning,
        MockDataCheckError,
        MockDataCheckErrorAndWarning,
    ]
    data_checks = DataChecks(data_checks=data_checks_list)
    assert data_checks.validate(X, y) == [
        DataCheckWarning(
            message="warning one",
            data_check_name="MockDataCheckWarning",
        ).to_dict(),
        DataCheckError(
            message="error one",
            data_check_name="MockDataCheckError",
        ).to_dict(),
        DataCheckWarning(
            message="warning two",
            data_check_name="MockDataCheckErrorAndWarning",
        ).to_dict(),
        DataCheckError(
            message="error two",
            data_check_name="MockDataCheckErrorAndWarning",
        ).to_dict(),
    ]


def get_expected_messages(problem_type):
    messages = [
        DataCheckWarning(
            message="Column(s) 'all_null', 'also_all_null' are 95.0% or more null",
            data_check_name="NullDataCheck",
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["all_null", "also_all_null"],
                "pct_null_rows": {"all_null": 1.0, "also_all_null": 1.0},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="NullDataCheck",
                    metadata={"columns": ["all_null", "also_all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'lots_of_null', 'nullable_integer', 'nullable_bool' have between 20.0% and 95.0% null values",
            data_check_name="NullDataCheck",
            message_code=DataCheckMessageCode.COLS_WITH_NULL,
            details={
                "columns": [
                    "lots_of_null",
                    "nullable_integer",
                    "nullable_bool",
                ],
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name="NullDataCheck",
                    metadata={
                        "columns": [
                            "lots_of_null",
                            "nullable_integer",
                            "nullable_bool",
                        ],
                        "is_target": False,
                    },
                    parameters={
                        "impute_strategies": {
                            "parameter_type": "column",
                            "columns": {
                                "lots_of_null": {
                                    "impute_strategy": {
                                        "categories": ["most_frequent"],
                                        "type": "category",
                                        "default_value": "most_frequent",
                                    },
                                },
                                "nullable_integer": {
                                    "impute_strategy": {
                                        "categories": ["mean", "most_frequent"],
                                        "type": "category",
                                        "default_value": "mean",
                                    },
                                },
                                "nullable_bool": {
                                    "impute_strategy": {
                                        "categories": ["most_frequent"],
                                        "type": "category",
                                        "default_value": "most_frequent",
                                    },
                                },
                            },
                        },
                    },
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Columns 'id' are 100.0% or more likely to be an ID column",
            data_check_name="IDColumnsDataCheck",
            message_code=DataCheckMessageCode.HAS_ID_COLUMN,
            details={"columns": ["id"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="IDColumnsDataCheck",
                    metadata={"columns": ["id"]},
                ),
            ],
        ).to_dict(),
        DataCheckError(
            message="1 row(s) (20.0%) of target values are null",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_HAS_NULL,
            details={"num_null_rows": 1, "pct_null_rows": 20.0},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name="InvalidTargetDataCheck",
                    parameters={
                        "impute_strategy": {
                            "parameter_type": DCAOParameterType.GLOBAL,
                            "type": "category",
                            "categories": ["mean", "most_frequent"]
                            if is_regression(problem_type)
                            else ["most_frequent"],
                            "default_value": "mean"
                            if is_regression(problem_type)
                            else "most_frequent",
                        },
                    },
                    metadata={"is_target": True},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="'all_null', 'also_all_null' has 0 unique values.",
            data_check_name="NoVarianceDataCheck",
            message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
            details={"columns": ["all_null", "also_all_null"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="NoVarianceDataCheck",
                    metadata={"columns": ["all_null", "also_all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="'lots_of_null' has 1 unique value.",
            data_check_name="NoVarianceDataCheck",
            message_code=DataCheckMessageCode.NO_VARIANCE,
            details={"columns": ["lots_of_null"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="NoVarianceDataCheck",
                    metadata={"columns": ["lots_of_null"]},
                ),
            ],
        ).to_dict(),
    ]
    return messages


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_classification(input_type, data_checks_input_dataframe):
    X = data_checks_input_dataframe

    y = pd.Series([0, 1, np.nan, 1, 0])
    y_multiclass = pd.Series([0, 1, np.nan, 2, 0])
    if input_type == "ww":
        y = ww.init_series(y)
        y_multiclass = ww.init_series(y_multiclass)

    data_checks = DefaultDataChecks(
        "binary",
        get_default_primary_search_objective("binary"),
    )
    imbalance = [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0, 1]",
            data_check_name="ClassImbalanceDataCheck",
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0.0, 1.0]},
        ).to_dict(),
    ]

    expected = get_expected_messages("binary")
    assert data_checks.validate(X, y) == expected + imbalance

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "binary",
                "objective": get_default_primary_search_objective("binary"),
            },
        },
    )
    assert data_checks.validate(X, y) == expected

    # multiclass
    imbalance = [
        DataCheckError(
            message="The number of instances of these targets is less than 2 * the number of cross folds = 6 instances: [0, 1, 2]",
            data_check_name="ClassImbalanceDataCheck",
            message_code=DataCheckMessageCode.CLASS_IMBALANCE_BELOW_FOLDS,
            details={"target_values": [0.0, 1.0, 2.0]},
        ).to_dict(),
    ]
    min_2_class_count = [
        DataCheckError(
            message="Target does not have at least two instances per class which is required for multiclass classification",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS,
            details={"least_populated_class_labels": [1.0, 2.0]},
        ).to_dict(),
    ]
    high_class_to_sample_ratio = [
        DataCheckWarning(
            message="Target has a large number of unique values, could be regression type problem.",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
            details={"class_to_value_ratio": 0.6},
        ).to_dict(),
    ]
    # multiclass
    data_checks = DefaultDataChecks(
        "multiclass",
        get_default_primary_search_objective("multiclass"),
    )

    expected = get_expected_messages("multiclass")

    assert (
        data_checks.validate(X, y_multiclass)
        == expected[:4]
        + min_2_class_count
        + high_class_to_sample_ratio
        + expected[4:]
        + imbalance
    )

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "multiclass",
                "objective": get_default_primary_search_objective("multiclass"),
            },
        },
    )
    assert (
        data_checks.validate(X, y_multiclass)
        == expected[:4] + min_2_class_count + high_class_to_sample_ratio + expected[4:]
    )


@pytest.mark.parametrize("input_type", ["pd", "ww"])
def test_default_data_checks_regression(input_type, data_checks_input_dataframe):
    X = data_checks_input_dataframe

    y = pd.Series([0.3, 100.0, np.nan, 1.0, 0.2])
    y_no_variance = pd.Series([5] * 5)
    if input_type == "ww":
        y = ww.init_series(y)
        y_no_variance = ww.init_series(y_no_variance)

    data_checks = DefaultDataChecks(
        "regression",
        get_default_primary_search_objective("regression"),
    )

    expected = get_expected_messages("regression")

    assert data_checks.validate(X, y) == expected

    # Skip Invalid Target
    assert (
        data_checks.validate(X, y_no_variance)
        == expected[:3]
        + expected[4:6]
        + [
            DataCheckWarning(
                message="Y has 1 unique value.",
                data_check_name="NoVarianceDataCheck",
                message_code=DataCheckMessageCode.NO_VARIANCE,
                details={"columns": ["Y"]},
            ).to_dict(),
        ]
        + expected[6:]
    )

    data_checks = DataChecks(
        DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES,
        {
            "InvalidTargetDataCheck": {
                "problem_type": "regression",
                "objective": get_default_primary_search_objective("regression"),
            },
        },
    )
    assert data_checks.validate(X, y) == expected


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
        },
    )
    y = pd.Series([0, 1, np.nan, 1, 0])
    data_checks = DefaultDataChecks(
        "regression",
        get_default_primary_search_objective("regression"),
    )
    highly_null_rows = SeriesWrap(pd.Series([1.0, 1.0, 1.0, 1.0, 1.0]))
    expected = [
        DataCheckWarning(
            message="5 out of 5 rows are 95.0% or more null",
            data_check_name="NullDataCheck",
            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            details={"pct_null_cols": highly_null_rows, "rows": [0, 1, 2, 3, 4]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_ROWS,
                    data_check_name="NullDataCheck",
                    metadata={"rows": [0, 1, 2, 3, 4]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="Column(s) 'all_null', 'also_all_null' are 95.0% or more null",
            data_check_name="NullDataCheck",
            message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
            details={
                "columns": ["all_null", "also_all_null"],
                "pct_null_rows": {"all_null": 1.0, "also_all_null": 1.0},
            },
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="NullDataCheck",
                    metadata={"columns": ["all_null", "also_all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckError(
            message="1 row(s) (20.0%) of target values are null",
            data_check_name="InvalidTargetDataCheck",
            message_code=DataCheckMessageCode.TARGET_HAS_NULL,
            details={"num_null_rows": 1, "pct_null_rows": 20.0},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name="InvalidTargetDataCheck",
                    parameters={
                        "impute_strategy": {
                            "parameter_type": DCAOParameterType.GLOBAL,
                            "type": "category",
                            "categories": ["mean", "most_frequent"],
                            "default_value": "mean",
                        },
                    },
                    metadata={"is_target": True},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="'all_null', 'also_all_null' has 0 unique values.",
            data_check_name="NoVarianceDataCheck",
            message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
            details={"columns": ["all_null", "also_all_null"]},
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="NoVarianceDataCheck",
                    metadata={"columns": ["all_null", "also_all_null"]},
                ),
            ],
        ).to_dict(),
        DataCheckError(
            message="Target is unsupported integer_nullable type. Valid Woodwork "
            "logical types include: integer, double",
            data_check_name="TargetDistributionDataCheck",
            message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
            details={"unsupported_type": "integer_nullable"},
            action_options=[],
        ).to_dict(),
    ]
    validation_messages = data_checks.validate(X, y)
    validation_messages[0]["details"]["pct_null_cols"] = SeriesWrap(
        validation_messages[0]["details"]["pct_null_cols"],
    )
    assert validation_messages == expected


@pytest.mark.parametrize("problem_type", ProblemTypes.all_problem_types)
def test_default_data_checks_across_problem_types(problem_type):
    default_data_check_list = DefaultDataChecks._DEFAULT_DATA_CHECK_CLASSES

    if is_time_series(problem_type):
        if is_classification(problem_type):
            default_data_check_list = default_data_check_list + [
                TimeSeriesSplittingDataCheck,
            ]
        default_data_check_list = default_data_check_list + [
            DateTimeFormatDataCheck,
            TimeSeriesParametersDataCheck,
        ]

    if problem_type in [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]:
        default_data_check_list = default_data_check_list + [
            TargetDistributionDataCheck,
        ]
    else:
        default_data_check_list = default_data_check_list + [ClassImbalanceDataCheck]

    problem_config = {
        "gap": 1,
        "max_delay": 1,
        "forecast_horizon": 1,
        "time_index": "datetime",
    }
    data_check_classes = [
        check.__class__
        for check in DefaultDataChecks(
            problem_type,
            get_default_primary_search_objective(problem_type),
            problem_configuration=problem_config
            if is_time_series(problem_type)
            else None,
        ).data_checks
    ]

    assert data_check_classes == default_data_check_list


def test_default_data_checks_missing_problem_configuration_for_time_series():
    with pytest.raises(
        ValueError,
        match="problem_configuration cannot be None for time series problems!",
    ):
        DefaultDataChecks(
            "time series binary",
            get_default_primary_search_objective("time series regression"),
        )


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
            r"Encountered the following error while initializing mock_check: .*__init__\(\) missing 1 required positional argument: 'bar'",
        ),
        (
            [MockCheck],
            {"mock_check": {"Bar": 2}},
            DataCheckInitError,
            r"Encountered the following error while initializing mock_check: .*__init__\(\) got an unexpected keyword argument 'Bar'",
        ),
        (
            [MockCheck],
            {"mock_check": {"fo": 3, "ba": 4}},
            DataCheckInitError,
            r"Encountered the following error while initializing mock_check: .*__init__\(\) got an unexpected keyword argument 'fo'",
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
    classes,
    params,
    expected_exception,
    expected_message,
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
    X, _, y = ts_data()
    y[0] = -1
    y = pd.Series(y)
    details = {"Count of offending values": sum(val <= 0 for val in y.values.flatten())}
    data_check_error = DataCheckError(
        message=f"Target has non-positive values which is not supported for {objective}",
        data_check_name="InvalidTargetDataCheck",
        message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
        details=details,
    ).to_dict()

    problem_config = {
        "gap": 1,
        "max_delay": 1,
        "forecast_horizon": 1,
        "time_index": "datetime",
    }
    default_data_check = DefaultDataChecks(
        problem_type="time series regression",
        objective=objective,
        problem_configuration=problem_config,
    ).data_checks
    for check in default_data_check:
        if check.name == "InvalidTargetDataCheck":
            assert check.validate(X, y) == [data_check_error]

    y = ww.init_series(y, logical_type="Categorical")
    default_data_check = DefaultDataChecks(
        problem_type="time series regression",
        objective=objective,
        problem_configuration=problem_config,
    ).data_checks
    data_check_error_type = DataCheckError(
        message=f"Target data type should be numeric for regression type problems.",
        data_check_name="InvalidTargetDataCheck",
        message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE_REGRESSION,
    ).to_dict()
    for check in default_data_check:
        if check.name == "InvalidTargetDataCheck":
            assert check.validate(X, y) == [data_check_error_type]


def test_data_checks_do_not_duplicate_actions(X_y_binary):
    X, y = X_y_binary

    class MockDataCheck(DataCheck):
        name = "Mock Data Check"

        def validate(self, X, y):
            return [
                DataCheckWarning(
                    message="warning one",
                    data_check_name=self.name,
                    message_code=None,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": ["col_to_drop"]},
                        ),
                    ],
                ).to_dict(),
            ]

    class MockDataCheckWithSameOutput(DataCheck):
        def validate(self, X, y):
            return [
                DataCheckWarning(
                    message="warning one",
                    data_check_name=self.name,
                    message_code=None,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": ["col_to_drop"]},
                        ),
                    ],
                ).to_dict(),
            ]

    data_checks_list = [MockDataCheck, MockDataCheckWithSameOutput]
    data_checks = DataChecks(data_checks=data_checks_list)

    assert data_checks.validate(X, y) == [
        DataCheckWarning(
            message="warning one",
            data_check_name="Mock Data Check",
            message_code=None,
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="Mock Data Check",
                    metadata={"columns": ["col_to_drop"]},
                ),
            ],
        ).to_dict(),
        DataCheckWarning(
            message="warning one",
            data_check_name="MockDataCheckWithSameOutput",
            message_code=None,
            action_options=[
                DataCheckActionOption(
                    DataCheckActionCode.DROP_COL,
                    data_check_name="MockDataCheckWithSameOutput",
                    metadata={"columns": ["col_to_drop"]},
                ),
            ],
        ).to_dict(),
    ]


def test_data_checks_drop_index(X_y_binary):
    X, y = X_y_binary
    X = pd.DataFrame(X)
    X["index_col"] = pd.Series(range(len(X)))
    X.ww.init(index="index_col")

    class MockDataCheck(DataCheck):
        def validate(self, X, y):
            return []

    assert MockDataCheck().validate(X, y) == []

    MockDataCheck.validate = MagicMock()
    checks = DataChecks([MockDataCheck, MockDataCheck, MockDataCheck])
    checks.validate(X, y)

    validate_args = MockDataCheck.validate.call_args_list
    for arg in validate_args:
        assert "index_col" not in arg[0][0].columns


def test_time_index_marked_as_sorted():
    X = pd.DataFrame()
    X["dates"] = [
        "1/1/21",
        "1/1/21",
        "1/7/21",
        "1/7/21",
        "1/7/21",
        "1/3/21",
        "1/10/21",
        "1/10/21",
        "1/4/21",
    ]
    y = pd.Series([i for i in range(9)])

    X_copy = X.copy()
    X_copy.ww.init(time_index="dates")
    dcs = DataChecks(
        [InvalidTargetDataCheck],
        {
            "InvalidTargetDataCheck": {
                "problem_type": "time series regression",
                "objective": get_default_primary_search_objective(
                    "time series regression",
                ),
            },
        },
    )
    results = dcs.validate(X_copy, y)
    assert len(results) == 1
    assert results[0]["code"] == "MISMATCHED_INDICES_ORDER"

    results = dcs.validate(X, y)
    assert not len(results)
