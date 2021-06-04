import numpy as np
import pandas as pd
import pytest

from evalml.data_checks import (
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
    NoVarianceDataCheck,
)

no_variance_data_check_name = NoVarianceDataCheck.name

all_distinct_X = pd.DataFrame({"feature": [1, 2, 3, 4]})
all_null_X = pd.DataFrame({"feature": [None] * 4, "feature_2": list(range(4))})
two_distinct_with_nulls_X = pd.DataFrame(
    {"feature": [1, 1, None, None], "feature_2": list(range(4))}
)
two_distinct_with_nulls_X_ww = two_distinct_with_nulls_X.copy()
two_distinct_with_nulls_X_ww.ww.init()

all_distinct_y = pd.Series([1, 2, 3, 4])
all_null_y = pd.Series([None] * 4)
two_distinct_with_nulls_y = pd.Series(([1] * 2) + ([None] * 2))
two_distinct_with_nulls_y_ww = two_distinct_with_nulls_y.copy()
two_distinct_with_nulls_y_ww.ww.init()
all_null_y_with_name = pd.Series([None] * 4)
all_null_y_with_name.name = "Labels"

feature_0_unique = DataCheckError(
    message="feature has 0 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"column": "feature"},
).to_dict()
feature_1_unique = DataCheckError(
    message="feature has 1 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"column": "feature"},
).to_dict()
labels_0_unique = DataCheckError(
    message="Y has 0 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"column": "Y"},
).to_dict()
labels_1_unique = DataCheckError(
    message="Y has 1 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"column": "Y"},
).to_dict()
drop_feature_action = DataCheckAction(
    DataCheckActionCode.DROP_COL, metadata={"column": "feature"}
).to_dict()

cases = [
    (
        all_distinct_X,
        all_distinct_y,
        True,
        {"warnings": [], "errors": [], "actions": []},
    ),
    (
        [[1], [2], [3], [4]],
        [1, 2, 3, 2],
        False,
        {"warnings": [], "errors": [], "actions": []},
    ),
    (
        np.arange(12).reshape(4, 3),
        [1, 2, 3],
        True,
        {"warnings": [], "errors": [], "actions": []},
    ),
    (
        all_null_X,
        all_distinct_y,
        False,
        {
            "warnings": [],
            "errors": [feature_0_unique],
            "actions": [drop_feature_action],
        },
    ),
    (
        all_null_X,
        [1] * 4,
        False,
        {
            "warnings": [],
            "errors": [feature_0_unique, labels_1_unique],
            "actions": [drop_feature_action],
        },
    ),
    (
        all_null_X,
        all_distinct_y,
        True,
        {
            "warnings": [],
            "errors": [feature_1_unique],
            "actions": [drop_feature_action],
        },
    ),
    (
        all_distinct_X,
        all_null_y,
        True,
        {"warnings": [], "errors": [labels_1_unique], "actions": []},
    ),
    (
        all_distinct_X,
        all_null_y,
        False,
        {"warnings": [], "errors": [labels_0_unique], "actions": []},
    ),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        True,
        {
            "warnings": [
                DataCheckWarning(
                    message="feature has two unique values including nulls. Consider encoding the nulls for "
                    "this column to be useful for machine learning.",
                    data_check_name=no_variance_data_check_name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"column": "feature"},
                ).to_dict(),
                DataCheckWarning(
                    message="Y has two unique values including nulls. Consider encoding the nulls for "
                    "this column to be useful for machine learning.",
                    data_check_name=no_variance_data_check_name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"column": "Y"},
                ).to_dict(),
            ],
            "errors": [],
            "actions": [drop_feature_action],
        },
    ),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        False,
        {
            "warnings": [],
            "errors": [feature_1_unique, labels_1_unique],
            "actions": [drop_feature_action],
        },
    ),
    (
        all_distinct_X,
        all_null_y_with_name,
        False,
        {
            "warnings": [],
            "errors": [
                DataCheckError(
                    message="Labels has 0 unique value.",
                    data_check_name=no_variance_data_check_name,
                    message_code=DataCheckMessageCode.NO_VARIANCE,
                    details={"column": "Labels"},
                ).to_dict()
            ],
            "actions": [],
        },
    ),
    (
        two_distinct_with_nulls_X_ww,
        two_distinct_with_nulls_y_ww,
        True,
        {
            "warnings": [
                DataCheckWarning(
                    message="feature has two unique values including nulls. Consider encoding the nulls for "
                    "this column to be useful for machine learning.",
                    data_check_name=no_variance_data_check_name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"column": "feature"},
                ).to_dict(),
                DataCheckWarning(
                    message="Y has two unique values including nulls. Consider encoding the nulls for "
                    "this column to be useful for machine learning.",
                    data_check_name=no_variance_data_check_name,
                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                    details={"column": "Y"},
                ).to_dict(),
            ],
            "errors": [],
            "actions": [drop_feature_action],
        },
    ),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        False,
        {
            "warnings": [],
            "errors": [feature_1_unique, labels_1_unique],
            "actions": [drop_feature_action],
        },
    ),
]


@pytest.mark.parametrize("X, y, count_nan_as_value, expected_validation_result", cases)
def test_no_variance_data_check_warnings(
    X, y, count_nan_as_value, expected_validation_result
):
    check = NoVarianceDataCheck(count_nan_as_value)
    assert check.validate(X, y) == expected_validation_result
