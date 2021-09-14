"""Data check that checks if the target data contains certain distributions that may need to be transformed prior training to improve model performance."""
import numpy as np
import woodwork as ww
from scipy.stats import shapiro

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class TargetDistributionDataCheck(DataCheck):
    """Check if the target data contains certain distributions that may need to be transformed prior training to improve model performance."""

    def validate(self, X, y):
        """Check if the target data has a certain distribution.

        Args:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the target data.

        Example:
            >>> from scipy.stats import lognorm
            >>> X = None
            >>> y = [0.946, 0.972, 1.154, 0.954, 0.969, 1.222, 1.038, 0.999, 0.973, 0.897]
            >>> target_check = TargetDistributionDataCheck()
            >>> assert target_check.validate(X, y) == {
            ...     "errors": [],
            ...     "warnings": [{"message": "Target may have a lognormal distribution.",
            ...                   "data_check_name": "TargetDistributionDataCheck",
            ...                   "level": "warning",
            ...                   "code": "TARGET_LOGNORMAL_DISTRIBUTION",
            ...                   "details": {"shapiro-statistic/pvalue": '0.84/0.045'}}],
            ...     "actions": [{'code': 'TRANSFORM_TARGET', 'metadata': {'column': None, 'transformation_strategy': 'lognormal', 'is_target': True}}]}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        if y is None:
            results["errors"].append(
                DataCheckError(
                    message="Target is None",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_IS_NONE,
                    details={},
                ).to_dict()
            )
            return results

        y = infer_feature_types(y)
        allowed_types = [
            ww.logical_types.Integer.type_string,
            ww.logical_types.Double.type_string,
        ]
        is_supported_type = y.ww.logical_type.type_string in allowed_types

        if not is_supported_type:
            results["errors"].append(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        y.ww.logical_type.type_string,
                        ", ".join([ltype for ltype in allowed_types]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict()
            )
            return results

        # Check if a normal distribution is detected with p-value above 0.05
        if shapiro(y).pvalue >= 0.05:
            return results

        y_new = round(y, 6)
        if any(y <= 0):
            y_new = y + abs(y.min()) + 1

        y_new = y_new[
            y_new < (y_new.mean() + 3 * round(y.std(), 3))
        ]  # Drop values greater than 3 standard deviations
        shapiro_test_og = shapiro(y_new)
        shapiro_test_log = shapiro(np.log(y_new))

        log_detected = False

        # If the p-value of the log transformed target is greater than or equal to the p-value of the original target
        # with outliers dropped, then it would imply that the log transformed target has more of a normal distribution
        if shapiro_test_log.pvalue >= shapiro_test_og.pvalue:
            log_detected = True

        if log_detected:
            details = {
                "shapiro-statistic/pvalue": f"{round(shapiro_test_og.statistic, 1)}/{round(shapiro_test_og.pvalue, 3)}"
            }
            results["warnings"].append(
                DataCheckWarning(
                    message="Target may have a lognormal distribution.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_LOGNORMAL_DISTRIBUTION,
                    details=details,
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.TRANSFORM_TARGET,
                    metadata={
                        "column": None,
                        "is_target": True,
                        "transformation_strategy": "lognormal",
                    },
                ).to_dict()
            )

        return results
