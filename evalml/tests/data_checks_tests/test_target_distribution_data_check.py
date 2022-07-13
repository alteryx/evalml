import numpy as np
import pandas as pd
import pytest
from scipy.stats import jarque_bera, lognorm, norm, shapiro

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
    TargetDistributionDataCheck,
)
from evalml.utils import infer_feature_types

target_dist_check_name = TargetDistributionDataCheck.name


def test_target_distribution_data_check_no_y(X_y_regression):
    X, y = X_y_regression
    y = None

    target_dist_check = TargetDistributionDataCheck()

    assert target_dist_check.validate(X, y) == [
        DataCheckError(
            message="Target is None",
            data_check_name=target_dist_check_name,
            message_code=DataCheckMessageCode.TARGET_IS_NONE,
            details={},
        ).to_dict(),
    ]


@pytest.mark.parametrize("target_type", ["boolean", "categorical", "integer", "double"])
def test_target_distribution_data_check_unsupported_target_type(target_type):
    X = pd.DataFrame(range(5))

    if target_type == "boolean":
        y = pd.Series([True, False] * 5)
    elif target_type == "categorical":
        y = pd.Series(["One", "Two", "Three", "Four", "Five"] * 2)
    elif target_type == "integer":
        y = [-1, -3, -5, 4, -2, 4, -4, 2, 1, 1]
    else:
        y = [9.2, 7.66, 4.93, 3.29, 4.06, -1.28, 4.95, 6.77, 9.07, 7.67]

    y = infer_feature_types(y)

    target_dist_check = TargetDistributionDataCheck()

    if target_type in ["integer", "double"]:
        assert target_dist_check.validate(X, y) == []
    else:
        assert target_dist_check.validate(X, y) == [
            DataCheckError(
                message=f"Target is unsupported {y.ww.logical_type.type_string} type. Valid Woodwork logical types include: integer, double",
                data_check_name=target_dist_check_name,
                message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                details={"unsupported_type": y.ww.logical_type.type_string},
            ).to_dict(),
        ]


@pytest.mark.parametrize("data_type", ["positive", "mixed", "negative"])
@pytest.mark.parametrize("distribution", ["normal", "lognormal", "very_lognormal"])
@pytest.mark.parametrize(
    "size,name,statistic",
    [(10000, "jarque_bera", jarque_bera), (5000, "shapiro", shapiro)],
)
def test_target_distribution_data_check_warning_action(
    size,
    name,
    statistic,
    distribution,
    data_type,
    X_y_regression,
):
    X, y = X_y_regression
    # set this to avoid flaky tests. This is primarily because when we have smaller samples,
    # once we remove values outside 3 st.devs, the distribution can begin to look more normal
    random_state = 2
    target_dist_check = TargetDistributionDataCheck()

    if distribution == "normal":
        y = norm.rvs(loc=3, size=size, random_state=random_state)
    elif distribution == "lognormal":
        y = lognorm.rvs(0.4, size=size, random_state=random_state)
    else:
        # Will have a p-value of 0 thereby rejecting the null hypothesis even after log transforming
        # This is essentially just checking the = of the statistic's "log.pvalue >= og.pvalue"
        y = lognorm.rvs(s=1, loc=1, scale=1, size=size, random_state=random_state)

    y = np.round(y, 6)

    if data_type == "negative":
        y = -np.abs(y)
    elif data_type == "mixed":
        y = y - 1.2

    if distribution == "normal":
        assert target_dist_check.validate(X, y) == []
    else:
        target_dist_ = target_dist_check.validate(X, y)

        if any(y <= 0):
            y = y + abs(y.min()) + 1
        y = y[y < (y.mean() + 3 * round(y.std(), 3))]
        test_og = statistic(y)

        details = {
            "normalization_method": name,
            "statistic": round(test_og.statistic, 1),
            "p-value": round(test_og.pvalue, 3),
        }
        assert target_dist_ == [
            DataCheckWarning(
                message="Target may have a lognormal distribution.",
                data_check_name=target_dist_check_name,
                message_code=DataCheckMessageCode.TARGET_LOGNORMAL_DISTRIBUTION,
                details=details,
                action_options=[
                    DataCheckActionOption(
                        DataCheckActionCode.TRANSFORM_TARGET,
                        data_check_name=target_dist_check_name,
                        metadata={
                            "is_target": True,
                            "transformation_strategy": "lognormal",
                        },
                    ),
                ],
            ).to_dict(),
        ]
