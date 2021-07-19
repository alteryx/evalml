import numpy as np
import pandas as pd
import pytest
from scipy.stats import lognorm, norm, shapiro

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
    TargetDistributionDataCheck,
)
from evalml.problem_types import handle_problem_types
from evalml.utils import infer_feature_types

target_dist_check_name = TargetDistributionDataCheck.name


def test_target_distribution_data_check_no_y(X_y_regression):
    X, y = X_y_regression
    y = None

    target_dist_check = TargetDistributionDataCheck("regression")

    assert target_dist_check.validate(X, y) == {
        "warnings": [],
        "errors": [
            DataCheckError(
                message="Target is None",
                data_check_name=target_dist_check_name,
                message_code=DataCheckMessageCode.TARGET_IS_NONE,
                details={},
            ).to_dict()
        ],
        "actions": [],
    }


@pytest.mark.parametrize("target_type", ["boolean", "categorical", "integer", "double"])
def test_target_distribution_data_check_unsupported_target_type(
    target_type, X_y_regression
):
    X, y = X_y_regression

    if target_type == "boolean":
        y = pd.Series([True, False] * 50)
    elif target_type == "categorical":
        y = pd.Series(["One", "Two", "Three", "Four"] * 25)
    elif target_type == "integer":
        y = pd.Series(np.random.randint(1, 25, 100))
    else:
        y = np.random.normal(4, 1, 100)

    y = infer_feature_types(y)

    target_dist_check = TargetDistributionDataCheck("regression")

    if target_type in ["integer", "double"]:
        assert target_dist_check.validate(X, y) == {
            "warnings": [],
            "errors": [],
            "actions": [],
        }
    else:
        assert target_dist_check.validate(X, y) == {
            "warnings": [],
            "errors": [
                DataCheckError(
                    message=f"Target is unsupported {y.ww.logical_type.type_string} type. Valid Woodwork logical types include: integer, double",
                    data_check_name=target_dist_check_name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict()
            ],
            "actions": [],
        }


@pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
def test_target_distribution_data_check_invalid_problem_type(
    problem_type, X_y_regression
):
    X, y = X_y_regression

    target_dist_check = TargetDistributionDataCheck(problem_type)

    if problem_type == "regression":
        assert target_dist_check.validate(X, y) == {
            "warnings": [],
            "errors": [],
            "actions": [],
        }
    else:
        assert target_dist_check.validate(X, y) == {
            "warnings": [],
            "errors": [
                DataCheckError(
                    message=f"Problem type {problem_type} is unsupported. Valid problem types include: [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]",
                    data_check_name=target_dist_check_name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_PROBLEM_TYPE,
                    details={
                        "unsupported_problem_type": handle_problem_types(problem_type)
                    },
                ).to_dict()
            ],
            "actions": [],
        }


@pytest.mark.parametrize("data_type", ["positive", "mixed", "negative"])
@pytest.mark.parametrize("distribution", ["normal", "lognormal", "lognormal_"])
def test_target_distribution_data_check_warning_action(
    distribution, data_type, X_y_regression
):
    X, y = X_y_regression

    target_dist_check = TargetDistributionDataCheck("regression")

    if distribution == "normal":
        y = norm.rvs(loc=3, size=10000)
    elif distribution == "lognormal":
        y = lognorm.rvs(0.4, size=10000)
    else:
        # Will have a p-value of 0 the null hypothesis even after log transforming
        y = lognorm.rvs(s=1, loc=1, scale=1, size=10000)

    if data_type == "negative":
        y = -np.abs(y)
    elif data_type == "mixed":
        y = y - 1.2

    if distribution == "normal":
        assert target_dist_check.validate(X, y) == {
            "warnings": [],
            "errors": [],
            "actions": [],
        }
    else:
        target_dist_ = target_dist_check.validate(X, y)

        if any(y <= 0):
            y = y + abs(y.min()) + 1
        y = y[y < (y.mean() + 3 * round(y.std(), 3))]
        shapiro_test_og = shapiro(y)

        details = {
            "shapiro-statistic/pvalue": f"{round(shapiro_test_og.statistic, 3)}/{round(shapiro_test_og.pvalue, 3)}"
        }
        assert target_dist_ == {
            "warnings": [
                DataCheckWarning(
                    message="Target may have a lognormal distribution.",
                    data_check_name=target_dist_check_name,
                    message_code=DataCheckMessageCode.TARGET_LOGNORMAL_DISTRIBUTION,
                    details=details,
                ).to_dict()
            ],
            "errors": [],
            "actions": [
                DataCheckAction(
                    DataCheckActionCode.TRANSFORM_TARGET,
                    metadata={
                        "column": None,
                        "is_target": True,
                        "transformation_strategy": "lognormal",
                    },
                ).to_dict()
            ],
        }
