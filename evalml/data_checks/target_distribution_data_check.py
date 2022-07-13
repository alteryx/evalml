"""Data check that checks if the target data contains certain distributions that may need to be transformed prior training to improve model performance."""
import numpy as np
import woodwork as ww
from scipy.stats import jarque_bera, shapiro

from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class TargetDistributionDataCheck(DataCheck):
    """Check if the target data contains certain distributions that may need to be transformed prior training to improve model performance. Uses the Shapiro-Wilks test when the dataset is <=5000 samples, otherwise uses Jarque-Bera."""

    def validate(self, X, y):
        """Check if the target data has a certain distribution.

        Args:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the target data.

        Examples:
            >>> import pandas as pd

            Targets that exhibit a lognormal distribution will raise a warning for the user to transform the target.

            >>> y = [0.946, 0.972, 1.154, 0.954, 0.969, 1.222, 1.038, 0.999, 0.973, 0.897]
            >>> target_check = TargetDistributionDataCheck()
            >>> assert target_check.validate(None, y) == [
            ...     {
            ...         "message": "Target may have a lognormal distribution.",
            ...         "data_check_name": "TargetDistributionDataCheck",
            ...         "level": "warning",
            ...         "code": "TARGET_LOGNORMAL_DISTRIBUTION",
            ...         "details": {"normalization_method": "shapiro", "statistic": 0.8, "p-value": 0.045, "columns": None, "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "TRANSFORM_TARGET",
            ...                 "data_check_name": "TargetDistributionDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {
            ...                     "transformation_strategy": "lognormal",
            ...                     "is_target": True,
            ...                     "columns": None,
            ...                     "rows": None
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]
            ...
            >>> y = pd.Series([1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5])
            >>> assert target_check.validate(None, y) == []
            ...
            ...
            >>> y = pd.Series(pd.date_range("1/1/21", periods=10))
            >>> assert target_check.validate(None, y) == [
            ...     {
            ...         "message": "Target is unsupported datetime type. Valid Woodwork logical types include: integer, double",
            ...         "data_check_name": "TargetDistributionDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None, "unsupported_type": "datetime"},
            ...         "code": "TARGET_UNSUPPORTED_TYPE",
            ...         "action_options": []
            ...     }
            ... ]
        """
        messages = []

        if y is None:
            messages.append(
                DataCheckError(
                    message="Target is None",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_IS_NONE,
                    details={},
                ).to_dict(),
            )
            return messages

        y = infer_feature_types(y)
        allowed_types = [
            ww.logical_types.Integer.type_string,
            ww.logical_types.Double.type_string,
        ]
        is_supported_type = y.ww.logical_type.type_string in allowed_types

        if not is_supported_type:
            messages.append(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        y.ww.logical_type.type_string,
                        ", ".join([ltype for ltype in allowed_types]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict(),
            )
            return messages

        (
            is_log_distribution,
            normalization_test_string,
            norm_test_og,
        ) = _detect_log_distribution_helper(y)
        if is_log_distribution:
            details = {
                "normalization_method": normalization_test_string,
                "statistic": round(norm_test_og.statistic, 1),
                "p-value": round(norm_test_og.pvalue, 3),
            }
            messages.append(
                DataCheckWarning(
                    message="Target may have a lognormal distribution.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_LOGNORMAL_DISTRIBUTION,
                    details=details,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.TRANSFORM_TARGET,
                            data_check_name=self.name,
                            metadata={
                                "is_target": True,
                                "transformation_strategy": "lognormal",
                            },
                        ),
                    ],
                ).to_dict(),
            )
        return messages


def _detect_log_distribution_helper(y):
    """Helper method to detect log distribution. Returns boolean, the normalization test used, and test statistics."""
    normalization_test = shapiro if len(y) <= 5000 else jarque_bera
    normalization_test_string = "shapiro" if len(y) <= 5000 else "jarque_bera"
    # Check if a normal distribution is detected with p-value above 0.05
    if normalization_test(y).pvalue >= 0.05:
        return False, normalization_test_string, None

    y_new = round(y, 6)
    if any(y <= 0):
        y_new = y + abs(y.min()) + 1
    y_new = y_new[
        y_new < (y_new.mean() + 3 * round(y.std(), 3))
    ]  # Drop values greater than 3 standard deviations
    norm_test_og = normalization_test(y_new)
    norm_test_log = normalization_test(np.log(y_new))

    # If the p-value of the log transformed target is greater than or equal to the p-value of the original target
    # with outliers dropped, then it would imply that the log transformed target has more of a normal distribution
    if norm_test_log.pvalue >= norm_test_og.pvalue:
        return True, normalization_test_string, norm_test_og
    return False, normalization_test_string, norm_test_og
