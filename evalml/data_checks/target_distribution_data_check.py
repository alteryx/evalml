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
    is_binary,
    is_multiclass,
    is_regression,
)
from evalml.utils.woodwork_utils import (
    infer_feature_types,
    numeric_and_boolean_ww,
)


class TargetDistributionDataCheck(DataCheck):
    """Checks if the target data contains certain distributions that may need to be transformed prior training to
     improve model performance."""

    def __init__(self, problem_type):
        """Check if the target is invalid for the specified problem type.

        Arguments:
            problem_type (str or ProblemTypes): The specific problem type to data check for.
                e.g. 'binary', 'multiclass', 'regression, 'time series regression'
        """
        self.problem_type = handle_problem_types(problem_type)
        raise ValueError("The target distribution data check doesn't support binary or multiclass problem types")

    def validate(self, X, y):
        """Checks if the target data has a certain distribution.

        Arguments:
            X (pd.DataFrame, np.ndarray): Features. Ignored.
            y (pd.Series, np.ndarray): Target data to check for underlying distributions.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if certain distributions are found in the target data.

        Example:
            >>> import pandas as pd
            >>> X = pd.DataFrame({"col": [1, 2, 3, 1]})
            >>> y = pd.Series([0, 1, None, None])
            >>> target_check = TargetDistributionDataCheck('regression')
            >>> assert target_check.validate(X, y) == {"errors": [{"message": "2 row(s) (50.0%) of target values are null",\
                                                                   "data_check_name": "TargetDistributionDataCheck",\
                                                                   "level": "error",\
                                                                   "code": "TARGET_HAS_NULL",\
                                                                   "details": {"num_null_rows": 2, "pct_null_rows": 50}}],\
                                                       "warnings": [],\
                                                       "actions": [{'code': 'IMPUTE_COL', 'metadata': {'column': None, 'impute_strategy': 'most_frequent', 'is_target': True}}]}
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
        is_supported_type = y.ww.logical_type in [
            ww.logical_types.Integer,
            ww.logical_types.Double
        ]

        if not is_supported_type:
            results["errors"].append(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        y.ww.logical_type,
                        ", ".join(
                            [ltype.type_string for ltype in numeric_and_boolean_ww]
                        ),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ).to_dict()
            )

        y_new = y + abs(y.min()) + 1

        ks_pvals = []
        for sigma in [0.1, 0.25, 0.3, 0.5, 0.7, 1.0, 1.5, 2]:
            dummy = lognorm.rvs(sigma, size=1000)
            ks_pval = ks_2samp(y_new, dummy, alternative="greater").pvalue
            ks_pvals.append(ks_pval)
        if sum(ks_pvals) == 8:
            details = {"kolomogoroc-smirnov-pvalues": ks_pvals}
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
                    DataCheckActionCode.TRANSFORM_TARGETZ,
                    metadata={
                        "column": None,
                        "is_target": True,
                        "impute_strategy": impute_strategy,
                    },
                ).to_dict()
            )






        null_rows = y.isnull()
        if null_rows.all():
            results["errors"].append(
                DataCheckError(
                    message="Target is either empty or fully null.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_IS_EMPTY_OR_FULLY_NULL,
                    details={},
                ).to_dict()
            )
            return results
        elif null_rows.any():
            num_null_rows = null_rows.sum()
            pct_null_rows = null_rows.mean() * 100
            results["errors"].append(
                DataCheckError(
                    message="{} row(s) ({}%) of target values are null".format(
                        num_null_rows, pct_null_rows
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                    details={
                        "num_null_rows": num_null_rows,
                        "pct_null_rows": pct_null_rows,
                    },
                ).to_dict()
            )
            impute_strategy = (
                "mean" if is_regression(self.problem_type) else "most_frequent"
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.IMPUTE_COL,
                    metadata={
                        "column": None,
                        "is_target": True,
                        "impute_strategy": impute_strategy,
                    },
                ).to_dict()
            )

        return results
