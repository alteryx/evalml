## Test data checks flow:
## User runs data checks. Gets actions as a result. Uses it to create components. (Eventually pass to AutoML.)
import pandas as pd
import woodwork as ww
from pandas.testing import assert_frame_equal, assert_series_equal

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
from evalml.data_checks.highly_null_data_check import HighlyNullDataCheck
from evalml.data_checks.invalid_targets_data_check import (
    InvalidTargetDataCheck,
)
from evalml.pipelines.components import DropColumns, TargetImputer
from evalml.pipelines.utils import _make_component_list_from_actions


def test_data_checks_with_healthy_data(X_y_binary):
    # Checks do not return any error.
    X, y = X_y_binary
    data_check = DefaultDataChecks(
        "binary", get_default_primary_search_objective("binary")
    )
    data_check_output = data_check.validate(X, y)
    assert data_check_output == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }
    assert _make_component_list_from_actions(data_check_output["actions"]) == []


def test_data_checks_return_drop_cols():
    X = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    y = pd.Series([1, 0, 0, 1, 1])
    data_check = HighlyNullDataCheck()
    data_checks_output = data_check.validate(X, y)
    assert data_checks_output == {
        "warnings": [
            DataCheckWarning(
                message="Columns 'all_null' are 95.0% or more null",
                data_check_name=HighlyNullDataCheck.name,
                message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                details={
                    "columns": ["all_null"],
                    "pct_null_rows": {"all_null": 1.0},
                    "null_row_indices": {"all_null": [0, 1, 2, 3, 4]},
                },
            ).to_dict()
        ],
        "errors": [],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.DROP_COL, metadata={"columns": ["all_null"]}
            ).to_dict(),
        ],
    }
    actions = [
        DataCheckAction.convert_dict_to_action(action)
        for action in data_checks_output["actions"]
    ]

    action_components = _make_component_list_from_actions(actions)
    assert action_components == [DropColumns(columns=["all_null"])]

    X_t = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "all_null": [None, None, None, None, None],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    X_expected = pd.DataFrame(
        {
            "lots_of_null": [None, 2, None, 3, 5],
            "no_null": [1, 2, 3, 4, 5],
        }
    )
    for component in action_components:
        X_t = component.fit_transform(X_t)
    assert_frame_equal(X_expected, X_t)


def test_data_checks_impute_cols():
    y = ww.init_series(pd.Series([0, 1, 1, None, None]))

    data_check = InvalidTargetDataCheck("binary", "Log Loss Binary")
    data_checks_output = data_check.validate(None, y)
    assert data_checks_output == {
        "warnings": [],
        "errors": [
            DataCheckError(
                message="2 row(s) (40.0%) of target values are null",
                data_check_name=InvalidTargetDataCheck.name,
                message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                details={"num_null_rows": 2, "pct_null_rows": 40.0},
            ).to_dict()
        ],
        "actions": [
            DataCheckAction(
                DataCheckActionCode.IMPUTE_COL,
                metadata={
                    "is_target": True,
                    "impute_strategy": "most_frequent",
                },
            ).to_dict()
        ],
    }
    actions = [
        DataCheckAction.convert_dict_to_action(action)
        for action in data_checks_output["actions"]
    ]

    action_components = _make_component_list_from_actions(actions)
    assert action_components == [
        TargetImputer(impute_strategy="most_frequent", fill_value=None)
    ]

    y_expected = ww.init_series(pd.Series([0, 1, 1, 1, 1]), logical_type="double")
    y_t = ww.init_series(pd.Series([0, 1, 1, None, None]))
    for component in action_components:
        _, y_t = component.fit_transform(None, y_t)
    assert_series_equal(y_expected, y_t)


def test_drop_rows():
    pass


def test_transform_target():
    pass
