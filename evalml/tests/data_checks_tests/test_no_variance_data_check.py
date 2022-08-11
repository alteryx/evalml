import numpy as np
import pandas as pd
import pytest
import woodwork as ww

from evalml.data_checks import (
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
    NoVarianceDataCheck,
)

no_variance_data_check_name = NoVarianceDataCheck.name

all_distinct_X = pd.DataFrame({"feature": [1, 2, 3, 4]})
all_null_X = pd.DataFrame({"feature": [None] * 4, "feature_2": list(range(4))})
two_distinct_with_nulls_X = pd.DataFrame(
    {"feature": [1, 1, None, None], "feature_2": list(range(4))},
)
two_distinct_with_nulls_X_ww = two_distinct_with_nulls_X.copy()
two_distinct_with_nulls_X_ww.ww.init()
two_distinct_with_nulls_X_ww_nullable_types = two_distinct_with_nulls_X.copy()
two_distinct_with_nulls_X_ww_nullable_types.ww.init(
    logical_types={"feature": "IntegerNullable"},
)

all_distinct_y = pd.Series([1, 2, 3, 4])
all_null_y = pd.Series([None] * 4)
two_distinct_with_nulls_y = pd.Series(([1] * 2) + ([None] * 2))
two_distinct_with_nulls_y_ww = two_distinct_with_nulls_y.copy()
two_distinct_with_nulls_y_ww = ww.init_series(two_distinct_with_nulls_y_ww)
two_distinct_with_nulls_y_ww_nullable_types = ww.init_series(
    two_distinct_with_nulls_y.copy(),
    logical_type="IntegerNullable",
)
all_null_y_with_name = pd.Series([None] * 4)
all_null_y_with_name.name = "Labels"

drop_feature_action_option = DataCheckActionOption(
    DataCheckActionCode.DROP_COL,
    data_check_name=no_variance_data_check_name,
    metadata={"columns": ["feature"]},
)
feature_0_unique = DataCheckWarning(
    message="'feature' has 0 unique values.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
    details={"columns": ["feature"]},
    action_options=[drop_feature_action_option],
).to_dict()
feature_1_unique = DataCheckWarning(
    message="'feature' has 1 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"columns": ["feature"]},
    action_options=[drop_feature_action_option],
).to_dict()
labels_0_unique = DataCheckWarning(
    message="Y has 0 unique values.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
    details={"columns": ["Y"]},
).to_dict()
labels_1_unique = DataCheckWarning(
    message="Y has 1 unique value.",
    data_check_name=no_variance_data_check_name,
    message_code=DataCheckMessageCode.NO_VARIANCE,
    details={"columns": ["Y"]},
).to_dict()


cases = [
    (
        all_distinct_X,
        all_distinct_y,
        True,
        [],
    ),
    (
        [[1], [2], [3], [4]],
        [1, 2, 3, 2],
        False,
        [],
    ),
    (
        np.arange(12).reshape(4, 3),
        [1, 2, 3],
        True,
        [],
    ),
    (all_null_X, all_distinct_y, False, [feature_0_unique]),
    (all_null_X, [1] * 4, False, [feature_0_unique, labels_1_unique]),
    (all_null_X, all_distinct_y, True, [feature_1_unique]),
    (all_distinct_X, all_null_y, True, [labels_1_unique]),
    (all_distinct_X, all_null_y, False, [labels_0_unique]),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        True,
        [
            DataCheckWarning(
                message="'feature' has two unique values including nulls. Consider encoding the nulls for "
                "this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["feature"]},
                action_options=[drop_feature_action_option],
            ).to_dict(),
            DataCheckWarning(
                message="Y has two unique values including nulls. Consider encoding the nulls for "
                "this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["Y"]},
            ).to_dict(),
        ],
    ),
    (
        two_distinct_with_nulls_X_ww_nullable_types,
        two_distinct_with_nulls_y_ww_nullable_types,
        True,
        [
            DataCheckWarning(
                message="'feature' has two unique values including nulls. Consider encoding the nulls for "
                "this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["feature"]},
                action_options=[drop_feature_action_option],
            ).to_dict(),
            DataCheckWarning(
                message="Y has two unique values including nulls. Consider encoding the nulls for "
                "this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["Y"]},
            ).to_dict(),
        ],
    ),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        False,
        [feature_1_unique, labels_1_unique],
    ),
    (
        all_distinct_X,
        all_null_y_with_name,
        False,
        [
            DataCheckWarning(
                message="Labels has 0 unique values.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_ZERO_UNIQUE,
                details={"columns": ["Labels"]},
            ).to_dict(),
        ],
    ),
    (
        two_distinct_with_nulls_X_ww,
        two_distinct_with_nulls_y_ww,
        True,
        [
            DataCheckWarning(
                message="'feature' has two unique values including nulls. Consider encoding the nulls for this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["feature"]},
                action_options=[drop_feature_action_option],
            ).to_dict(),
            DataCheckWarning(
                message="Y has two unique values including nulls. Consider encoding the nulls for this column to be useful for machine learning.",
                data_check_name=no_variance_data_check_name,
                message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                details={"columns": ["Y"]},
            ).to_dict(),
        ],
    ),
    (
        two_distinct_with_nulls_X,
        two_distinct_with_nulls_y,
        False,
        [feature_1_unique, labels_1_unique],
    ),
    (
        two_distinct_with_nulls_X,
        None,
        False,
        [feature_1_unique],
    ),
]


@pytest.mark.parametrize("X, y, count_nan_as_value, expected_validation_result", cases)
def test_no_variance_data_check_warnings(
    X,
    y,
    count_nan_as_value,
    expected_validation_result,
):
    check = NoVarianceDataCheck(count_nan_as_value)
    assert check.validate(X, y) == expected_validation_result
