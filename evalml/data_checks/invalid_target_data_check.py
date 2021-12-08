"""Data check that checks if the target data contains missing or invalid values."""
import woodwork as ww

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.objectives import get_objective
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


class InvalidTargetDataCheck(DataCheck):
    """Check if the target data is considered invalid.

    Target data is considered invalid if:
        - Target is None.
        - Target has NaN or None values.
        - Target is of an unsupported Woodwork logical type.
        - Target and features have different lengths or indices.
        - Target does not have enough instances of a class in a classification problem.
        - Target does not contain numeric data for regression problems.

    Args:
        problem_type (str or ProblemTypes): The specific problem type to data check for.
            e.g. 'binary', 'multiclass', 'regression, 'time series regression'
        objective (str or ObjectiveBase): Name or instance of the objective class.
        n_unique (int): Number of unique target values to store when problem type is binary and target
            incorrectly has more than 2 unique values. Non-negative integer. If None, stores all unique values. Defaults to 100.
    """

    multiclass_continuous_threshold = 0.05

    def __init__(self, problem_type, objective, n_unique=100):
        self.problem_type = handle_problem_types(problem_type)
        self.objective = get_objective(objective)
        if n_unique is not None and n_unique <= 0:
            raise ValueError("`n_unique` must be a non-negative integer value.")
        self.n_unique = n_unique

    def validate(self, X, y):
        """Check if the target data is considered invalid. If the input features argument is not None, it will be used to check that the target and features have the same dimensions and indices.

        Target data is considered invalid if:
            - Target is None.
            - Target has NaN or None values.
            - Target is of an unsupported Woodwork logical type.
            - Target and features have different lengths or indices.
            - Target does not have enough instances of a class in a classification problem.
            - Target does not contain numeric data for regression problems.

        Args:
            X (pd.DataFrame, np.ndarray): Features. If not None, will be used to check that the target and features have the same dimensions and indices.
            y (pd.Series, np.ndarray): Target data to check for invalid values.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if any invalid values are found in the target data.

        Examples:
            >>> import pandas as pd
            ...
            >>> X = pd.DataFrame({"col": [1, 2, 3, 1]})
            >>> y = pd.Series(["cat_1", "cat_2", "cat_1", "cat_2"])
            >>> target_check = InvalidTargetDataCheck('regression', 'R2')
            >>> assert target_check.validate(X, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Target is unsupported Unknown type. Valid Woodwork logical types include: integer, double, boolean',
            ...                 'data_check_name': 'InvalidTargetDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None, 'unsupported_type': 'unknown'},
            ...                 'code': 'TARGET_UNSUPPORTED_TYPE'},
            ...                {'message': 'Target data type should be numeric for regression type problems.',
            ...                 'data_check_name': 'InvalidTargetDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None},
            ...                 'code': 'TARGET_UNSUPPORTED_TYPE'}],
            ...     'actions': []}
            ...
            ...
            >>> y = pd.Series([None, pd.NA, pd.NaT, None])
            >>> assert target_check.validate(X, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Target is either empty or fully null.',
            ...                 'data_check_name': 'InvalidTargetDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None},
            ...                 'code': 'TARGET_IS_EMPTY_OR_FULLY_NULL'}],
            ...     'actions': []}
            ...
            ...
            >>> y = pd.Series([1, None, 3, None])
            >>> assert target_check.validate(None, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': '2 row(s) (50.0%) of target values are null',
            ...                 'data_check_name': 'InvalidTargetDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None,
            ...                             'rows': None,
            ...                             'num_null_rows': 2,
            ...                             'pct_null_rows': 50.0},
            ...                 'code': 'TARGET_HAS_NULL'}],
            ...     'actions': [{'code': 'IMPUTE_COL',
            ...                  'data_check_name': 'InvalidTargetDataCheck',
            ...                  'metadata': {'columns': None,
            ...                               'rows': None,
            ...                               'is_target': True,
            ...                               'impute_strategy': 'mean'}}]}
            ...
            ...
            >>> X = pd.DataFrame([i for i in range(50)])
            >>> y = pd.Series([i%2 for i in range(50)])
            >>> target_check = InvalidTargetDataCheck('multiclass', 'Log Loss Multiclass')
            >>> assert target_check.validate(X, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Target has two or less classes, which is too few for multiclass problems.  Consider changing to binary.',
            ...                 'data_check_name': 'InvalidTargetDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None, 'num_classes': 2},
            ...                 'code': 'TARGET_MULTICLASS_NOT_ENOUGH_CLASSES'}],
            ...     'actions': []}
            ...
            ...
            >>> target_check = InvalidTargetDataCheck('regression', 'R2')
            >>> X = pd.DataFrame([i for i in range(5)])
            >>> y = pd.Series([1, 2, 4, 3], index=[1, 2, 4, 3])
            >>> assert target_check.validate(X, y) == {
            ...     'warnings': [{'message': 'Input target and features have different lengths',
            ...                   'data_check_name': 'InvalidTargetDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': None,
            ...                               'rows': None,
            ...                               'features_length': 5,
            ...                               'target_length': 4},
            ...                   'code': 'MISMATCHED_LENGTHS'},
            ...                  {'message': 'Input target and features have mismatched indices. Details will include the first 10 mismatched indices.',
            ...                   'data_check_name': 'InvalidTargetDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': None,
            ...                               'rows': None,
            ...                               'indices_not_in_features': [],
            ...                               'indices_not_in_target': [0]},
            ...                   'code': 'MISMATCHED_INDICES'}],
            ...     'errors': [],
            ...     'actions': []}
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
        results = self._check_target_has_nan(y, results)
        if any(
            error["code"] == "TARGET_IS_EMPTY_OR_FULLY_NULL"
            for error in results["errors"]
        ):
            # If our target is empty or fully null, no need to check for other invalid targets, return immediately.
            return results
        results = self._check_target_is_unsupported_type(y, results)
        results = self._check_regression_target(y, results)
        results = self._check_classification_target(y, results)
        results = self._check_for_non_positive_target(y, results)
        results = self._check_target_and_features_compatible(X, y, results)
        return results

    def _check_target_is_unsupported_type(self, y, results):
        is_supported_type = y.ww.logical_type.type_string in numeric_and_boolean_ww + [
            ww.logical_types.Categorical.type_string,
        ]
        if not is_supported_type:
            DataCheck._add_message(
                DataCheckError(
                    message="Target is unsupported {} type. Valid Woodwork logical types include: {}".format(
                        type(y.ww.logical_type),
                        ", ".join([ltype for ltype in numeric_and_boolean_ww]),
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={"unsupported_type": y.ww.logical_type.type_string},
                ),
                results,
            )
        return results

    def _check_target_has_nan(self, y, results):
        null_rows = y.isnull()
        if null_rows.all():
            DataCheck._add_message(
                DataCheckError(
                    message="Target is either empty or fully null.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_IS_EMPTY_OR_FULLY_NULL,
                    details={},
                ),
                results,
            )
            return results
        elif null_rows.any():
            num_null_rows = null_rows.sum()
            pct_null_rows = null_rows.mean() * 100
            DataCheck._add_message(
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
                ),
                results,
            )
            impute_strategy = (
                "mean" if is_regression(self.problem_type) else "most_frequent"
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.IMPUTE_COL,
                    data_check_name=self.name,
                    metadata={
                        "is_target": True,
                        "impute_strategy": impute_strategy,
                    },
                ).to_dict()
            )
        return results

    def _check_target_and_features_compatible(self, X, y, results):
        if X is not None:
            X = infer_feature_types(X)
            X_index = list(X.index)
            y_index = list(y.index)
            X_length = len(X_index)
            y_length = len(y_index)
            if X_length != y_length:
                DataCheck._add_message(
                    DataCheckWarning(
                        message="Input target and features have different lengths",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.MISMATCHED_LENGTHS,
                        details={
                            "features_length": X_length,
                            "target_length": y_length,
                        },
                    ),
                    results,
                )

            if X_index != y_index:
                if set(X_index) == set(y_index):
                    DataCheck._add_message(
                        DataCheckWarning(
                            message="Input target and features have mismatched indices order.",
                            data_check_name=self.name,
                            message_code=DataCheckMessageCode.MISMATCHED_INDICES_ORDER,
                            details={},
                        ),
                        results,
                    )
                else:
                    index_diff_not_in_X = list(set(y_index) - set(X_index))[:10]
                    index_diff_not_in_y = list(set(X_index) - set(y_index))[:10]
                    DataCheck._add_message(
                        DataCheckWarning(
                            message="Input target and features have mismatched indices. Details will include the first 10 mismatched indices.",
                            data_check_name=self.name,
                            message_code=DataCheckMessageCode.MISMATCHED_INDICES,
                            details={
                                "indices_not_in_features": index_diff_not_in_X,
                                "indices_not_in_target": index_diff_not_in_y,
                            },
                        ),
                        results,
                    )
        return results

    def _check_for_non_positive_target(self, y, results):
        any_neg = (
            not (y > 0).all()
            if y.ww.logical_type.type_string
            in [
                ww.logical_types.Integer.type_string,
                ww.logical_types.Double.type_string,
            ]
            else None
        )
        if any_neg and self.objective.positive_only:
            details = {
                "Count of offending values": sum(val <= 0 for val in y.values.flatten())
            }
            DataCheck._add_message(
                DataCheckError(
                    message=f"Target has non-positive values which is not supported for {self.objective.name}",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_INCOMPATIBLE_OBJECTIVE,
                    details=details,
                ),
                results,
            )
        return results

    def _check_regression_target(self, y, results):
        if (
            self.problem_type == ProblemTypes.REGRESSION
            and "numeric" not in y.ww.semantic_tags
        ):
            DataCheck._add_message(
                DataCheckError(
                    message="Target data type should be numeric for regression type problems.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                    details={},
                ),
                results,
            )
        return results

    def _check_classification_target(self, y, results):
        value_counts = y.value_counts()
        unique_values = value_counts.index.tolist()

        if is_binary(self.problem_type) and len(value_counts) != 2:
            if self.n_unique is None:
                details = {"target_values": unique_values}
            else:
                details = {
                    "target_values": unique_values[
                        : min(self.n_unique, len(unique_values))
                    ]
                }
            DataCheck._add_message(
                DataCheckError(
                    message="Binary class targets require exactly two unique values.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                    details=details,
                ),
                results,
            )
        elif is_multiclass(self.problem_type):
            if value_counts.min() <= 1:
                least_populated = value_counts[value_counts <= 1]
                details = {
                    "least_populated_class_labels": sorted(
                        least_populated.index.tolist()
                    )
                }
                DataCheck._add_message(
                    DataCheckError(
                        message="Target does not have at least two instances per class which is required for multiclass classification",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_TWO_EXAMPLES_PER_CLASS,
                        details=details,
                    ),
                    results,
                )
            if len(unique_values) <= 2:
                details = {"num_classes": len(unique_values)}
                DataCheck._add_message(
                    DataCheckError(
                        message="Target has two or less classes, which is too few for multiclass problems.  Consider changing to binary.",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.TARGET_MULTICLASS_NOT_ENOUGH_CLASSES,
                        details=details,
                    ),
                    results,
                )

            num_class_to_num_value_ratio = len(unique_values) / len(y)
            if num_class_to_num_value_ratio >= self.multiclass_continuous_threshold:
                details = {"class_to_value_ratio": num_class_to_num_value_ratio}
                DataCheck._add_message(
                    DataCheckWarning(
                        message="Target has a large number of unique values, could be regression type problem.",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.TARGET_MULTICLASS_HIGH_UNIQUE_CLASS,
                        details=details,
                    ),
                    results,
                )
        return results
