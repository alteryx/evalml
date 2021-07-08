import numpy as np
from scipy.stats import ks_2samp, lognorm
import woodwork as ww

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
)
from evalml.utils.woodwork_utils import (
    infer_feature_types,
)


class TargetDistributionDataCheck(DataCheck):
    """Checks if the target data contains certain distributions that may need to be transformed prior training to
     improve model performance."""

    def __init__(self, problem_type):
        """Check if the target is invalid for the specified problem type.

        Arguments:
            problem_type (str or ProblemTypes): The specific problem type to data check for.
                e.g. 'regression, 'time series regression'
        """
        self.problem_type = handle_problem_types(problem_type)

    def validate(self, X, y):
        """Checks if the target data has a certain distribution.

        Arguments:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the target data.

        Example:
            >>> from scipy.stats import lognorm
            >>> X = None
            >>> y = lognorm.rvs(0.1, size=1000)
            >>> target_check = TargetDistributionDataCheck('regression')
            >>> assert target_check.validate(X, y) == {"errors": [],\
                                                       "warnings": [{"message": "Target may have a lognormal distribution.",\
                                                                    "data_check_name": "TargetDistributionDataCheck",\
                                                                    "level": "warning",\
                                                                    "code": "TARGET_LOGNORMAL_DISTRIBUTION",\
                                                                    "details": {"kolomogoroc-smirnov-sigma-pvalues": [(0.1, 1.0), (0.25, 1.0), (0.3, 1.0), (0.5, 1.0), (0.7, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0)]}}],\
                                                       "actions": [{'code': 'TRANSFORM_TARGET', 'metadata': {'column': None, 'transformation_strategy': 'lognormal', 'is_target': True}}]}
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

        if self.problem_type not in [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]:
            results["errors"].append(
                DataCheckError(
                    message="Problem type {} is unsupported. Valid problem types include: [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]".format(
                        self.problem_type,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_PROBLEM_TYPE,
                    details={"unsupported_problem_type": self.problem_type},
                ).to_dict()
            )
            return results

        y = infer_feature_types(y)
        allowed_types = [
            ww.logical_types.Integer.type_string,
            ww.logical_types.Double.type_string
        ]
        is_supported_type = y.ww.logical_type.type_string in allowed_types

        if not is_supported_type:
            results["errors"].append(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        y.ww.logical_type.type_string,
                        ", ".join(
                            [ltype for ltype in allowed_types]
                        ),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict()
            )
            return results

        if any(y <= 0):
            y_new = y + abs(y.min()) + 1
        else:
            y_new = y

        ks_pvals = []
        for sigma in [0.7, 0.85, 1.0, 1.25, 1.5, 2]:
            dummy = lognorm.rvs(sigma, size=1000)
            ks_pval = ks_2samp(y_new, dummy, alternative="greater").pvalue
            ks_pvals.append((sigma, ks_pval))
        print("----------------------")
        print(ks_pvals)
        if sum(np.array(ks_pvals))[1] == 6:
            details = {"kolomogoroc-smirnov-sigma-pvalues": ks_pvals}
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
