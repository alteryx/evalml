## Test data checks flow:
## User runs data checks. Gets actions as a result. Uses it to create components. (Eventually pass to AutoML.)
import pandas as pd

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
from evalml.pipelines.components import DropColumns
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


def test_return_row_removal():
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

    assert _make_component_list_from_actions(actions) == [
        DropColumns(columns=["all_null"])
    ]


def test_impute_col():
    pass


def test_transform_target():
    pass


def test_drop_col():
    pass
