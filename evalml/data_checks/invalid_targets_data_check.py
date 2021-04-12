import woodwork as ww

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.objectives import get_objective
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_binary,
    is_multiclass,
    is_regression
)
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types,
    numeric_and_boolean_ww
)


class InvalidTargetDataCheck(DataCheck):
    """Checks if the target data contains missing or invalid values."""

    multiclass_continuous_threshold = .05

    def __init__(self, problem_type, objective, n_unique=100):
        """Check if the target is invalid for the specified problem type.

        Arguments:
            problem_type (str or ProblemTypes): The specific problem type to data check for.
                e.g. 'binary', 'multiclass', 'regression, 'time series regression'
            objective (str or ObjectiveBase): Name or instance of the objective class.
            n_unique (int): Number of unique target values to store when problem type is binary and target
                incorrectly has more than 2 unique values. Non-negative integer. Defaults to 100. If None, stores all unique values.
        """
        self.problem_type = handle_problem_types(problem_type)
        self.objective = get_objective(objective)
        if n_unique is not None and n_unique <= 0:
            raise ValueError("`n_unique` must be a non-negative integer value.")
        self.n_unique = n_unique

    def validate(self, X, y):
        """Checks if the target data contains missing or invalid values.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features. Ignored.
            y (ww.DataColumn, pd.Series, np.ndarray): Target data to check for invalid values.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if any invalid values are found in the target data.

        Example:
            >>> import pandas as pd
            >>> X = pd.DataFrame({"col": [1, 2, 3, 1]})
            >>> y = pd.Series([0, 1, None, None])
            >>> target_check = InvalidTargetDataCheck('binary', 'Log Loss Binary')
            >>> assert target_check.validate(X, y) == {"errors": [{"message": "2 row(s) (50.0%) of target values are null",\
                                                                   "data_check_name": "InvalidTargetDataCheck",\
                                                                   "level": "error",\
                                                                   "code": "TARGET_HAS_NULL",\
                                                                   "details": {"num_null_rows": 2, "pct_null_rows": 50}}],\
                                                       "warnings": [],\
                                                       "actions": [{'code': 'IMPUTE_COL', 'metadata': {'column': None, 'impute_strategy': 'most_frequent', 'is_target': True}}]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        if y is None:
            results["errors"].append(DataCheckError(message="Target is None",
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_IS_NONE,
                                                    details={}).to_dict())
            return results

        y = infer_feature_types(y)
        is_supported_type = y.logical_type in numeric_and_boolean_ww + [ww.logical_types.Categorical]
        if not is_supported_type:
            results["errors"].append(DataCheckError(message="Target is unsupported {} type. Valid Woodwork logical types include: {}"
                                                    .format(y.logical_type, ", ".join([ltype.type_string for ltype in numeric_and_boolean_ww])),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                                                    details={"unsupported_type": y.logical_type.type_string}).to_dict())
        y_df = _convert_woodwork_types_wrapper(y.to_series())
        null_rows = y_df.isnull()
        if null_rows.all():
            results["errors"].append(DataCheckError(message="Target is either empty or fully null.",
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_IS_EMPTY_OR_FULLY_NULL,
                                                    details={}).to_dict())
            return results
        elif null_rows.any():
            num_null_rows = null_rows.sum()
            pct_null_rows = null_rows.mean() * 100
            results["errors"].append(DataCheckError(message="{} row(s) ({}%) of target values are null".format(num_null_rows, pct_null_rows),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                                    details={"num_null_rows": num_null_rows, "pct_null_rows": pct_null_rows}).to_dict())
            impute_strategy = "mean" if is_regression(self.problem_type) else "most_frequent"
            results["actions"].append(DataCheckAction(DataCheckActionCode.IMPUTE_COL,
                                                      metadata={"column": None, "is_target": True, "impute_strategy": impute_strategy}).to_dict())

        value_counts = y_df.value_counts()
        unique_values = value_counts.index.tolist()

        if is_binary(self.problem_type) and len(value_counts) != 2:
            if self.n_unique is None:
                details = {"target_values": unique_values}
            else:
                details = {"target_values": unique_values[:min(self.n_unique, len(unique_values))]}
            results["errors"].append(DataCheckError(message="Binary class targets require exactly two unique values.",
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                                    details=details).to_dict())

        if self.problem_type == ProblemTypes.REGRESSION and "numeric" not in y.semantic_tags:
            results["errors"].append(DataCheckError(message="Target data type should be numeric for regression type problems.",
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                                                    details={}).to_dict())

        if is_multiclass(self.problem_type):
            if value_counts.min() <= 1:
                least_populated = value_counts[value_counts <= 1]
                details = {"least_populated_class_labels": least_populated.index.tolist()}
                results["errors"].append(DataCheckError(message="Target does not have at least two instances per class which is required for multiclass classification",
                                                        data_check_name=self.name,
                                                        message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS,
                                                        details=details).to_dict())
            if len(unique_values) <= 2:
                details = {"num_classes": len(unique_values)}
                results["errors"].append(DataCheckError(
                    message="Target has two or less classes, which is too few for multiclass problems.  Consider changing to binary.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_ENOUGH_CLASSES,
                    details=details).to_dict())

            num_class_to_num_value_ratio = len(unique_values) / len(y)
            if num_class_to_num_value_ratio >= self.multiclass_continuous_threshold:
                details = {"class_to_value_ratio": num_class_to_num_value_ratio}
                results["warnings"].append(DataCheckWarning(
                    message="Target has a large number of unique values, could be regression type problem.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
                    details=details).to_dict())

        any_neg = not (y_df > 0).all() if y.logical_type in [ww.logical_types.Integer, ww.logical_types.Double] else None
        if any_neg and self.objective.positive_only:
            details = {"Count of offending values": sum(val <= 0 for val in y_df.values.flatten())}
            results["errors"].append(DataCheckError(message=f"Target has non-positive values which is not supported for {self.objective.name}",
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
                                                    details=details).to_dict())

        if X is not None:
            X = infer_feature_types(X)
            X_index = list(X.to_dataframe().index)
            y_index = list(y_df.index)
            X_length = len(X_index)
            y_length = len(y_index)
            if X_length != y_length:
                results["warnings"].append(DataCheckWarning(message="Input target and features have different lengths",
                                                            data_check_name=self.name,
                                                            message_code=DataCheckMessageCode.MISMATCHED_LENGTHS,
                                                            details={"features_length": X_length, "target_length": y_length}).to_dict())

            if X_index != y_index:
                if set(X_index) == set(y_index):
                    results["warnings"].append(DataCheckWarning(message="Input target and features have mismatched indices order",
                                                                data_check_name=self.name,
                                                                message_code=DataCheckMessageCode.MISMATCHED_INDICES_ORDER,
                                                                details={}).to_dict())
                else:
                    index_diff_not_in_X = list(set(y_index) - set(X_index))[:10]
                    index_diff_not_in_y = list(set(X_index) - set(y_index))[:10]
                    results["warnings"].append(DataCheckWarning(message="Input target and features have mismatched indices",
                                                                data_check_name=self.name,
                                                                message_code=DataCheckMessageCode.MISMATCHED_INDICES,
                                                                details={"indices_not_in_features": index_diff_not_in_X,
                                                                         "indices_not_in_target": index_diff_not_in_y}).to_dict())

        return results
