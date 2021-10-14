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
from evalml.pipelines.utils import _make_component_list_from_actions


def test_data_checks_with_healthy_data(X_y_binary):
    # Checks do not return any error.
    X, y = X_y_binary
    data_checks = DefaultDataChecks(
        "binary", get_default_primary_search_objective("binary")
    )
    data_checks_output = data_checks.validate(X, y)
    assert data_checks_output == {
        "warnings": [],
        "errors": [],
        "actions": [],
    }
    assert _make_component_list_from_actions(data_checks_output["actions"]) == []


def test_return_row_removal():
    pass


def test_impute_col():
    pass


def test_transform_target():
    pass


def test_drop_col():
    pass
