"""Data check that checks if there are any columns in the input that are either too unique for classification problems or not unique enough for regression problems."""
from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.problem_types import (
    handle_problem_types,
    is_multiclass,
    is_regression,
)
from evalml.utils.woodwork_utils import infer_feature_types

warning_not_unique_enough = (
    "Input columns {} for {} problem type are not unique enough."
)
warning_too_unique = "Input columns {} for {} problem type are too unique."


class UniquenessDataCheck(DataCheck):
    """Check if there are any columns in the input that are either too unique for classification problems or not unique enough for regression problems.

    Args:
        problem_type (str or ProblemTypes): The specific problem type to data check for.
            e.g. 'binary', 'multiclass', 'regression, 'time series regression'
        threshold(float): The threshold to set as an upper bound on uniqueness for classification type problems
            or lower bound on for regression type problems.  Defaults to 0.50.
    """

    def __init__(self, problem_type, threshold=0.50):
        self.problem_type = handle_problem_types(problem_type)
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Check if there are any columns in the input that are too unique in the case of classification problems or not unique enough in the case of regression problems.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored.  Defaults to None.

        Returns:
            dict: dict with a DataCheckWarning if there are any too unique or not
                unique enough columns.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...    'regression_unique_enough': [float(x) for x in range(100)],
            ...    'regression_not_unique_enough': [float(1) for x in range(100)]
            ... })
            >>> uniqueness_check = UniquenessDataCheck(problem_type="regression", threshold=0.8)
            >>> assert uniqueness_check.validate(df) == {
            ...     "errors": [],
            ...     "warnings": [{"message": "Input columns 'regression_not_unique_enough' for regression problem type are not unique enough.",
            ...                   "data_check_name": "UniquenessDataCheck",
            ...                   "level": "warning",
            ...                   "code": "NOT_UNIQUE_ENOUGH",
            ...                   "details": {"columns": ["regression_not_unique_enough"], "uniqueness_score": {"regression_not_unique_enough": 0.0}, "rows": None}}],
            ...     "actions": [{"code": "DROP_COL",
            ...                  "metadata": {"columns": ["regression_not_unique_enough"], "rows": None}}]}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)

        res = X.apply(UniquenessDataCheck.uniqueness_score)

        if is_regression(self.problem_type):
            not_unique_enough_cols = list(res.index[res < self.threshold])
            results["warnings"].append(
                DataCheckWarning(
                    message=warning_not_unique_enough.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in not_unique_enough_cols]
                        ),
                        self.problem_type,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.NOT_UNIQUE_ENOUGH,
                    details={
                        "columns": not_unique_enough_cols,
                        "uniqueness_score": {
                            col: res.loc[col] for col in not_unique_enough_cols
                        },
                    },
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    action_code=DataCheckActionCode.DROP_COL,
                    metadata={"columns": not_unique_enough_cols},
                ).to_dict()
            )
        elif is_multiclass(self.problem_type):
            too_unique_cols = list(res.index[res > self.threshold])
            results["warnings"].append(
                DataCheckWarning(
                    message=warning_too_unique.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in too_unique_cols]
                        ),
                        self.problem_type,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TOO_UNIQUE,
                    details={
                        "columns": too_unique_cols,
                        "uniqueness_score": {
                            col: res.loc[col] for col in too_unique_cols
                        },
                    },
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    action_code=DataCheckActionCode.DROP_COL,
                    metadata={"columns": too_unique_cols},
                ).to_dict()
            )
        return results

    @staticmethod
    def uniqueness_score(col):
        """Calculate a uniqueness score for the provided field.  NaN values are not considered as unique values in the calculation.

        Based on the Herfindahl–Hirschman Index.

        Args:
            col (pd.Series): Feature values.

        Returns:
            (float): Uniqueness score.
        """
        norm_counts = col.value_counts() / col.value_counts().sum()
        square_counts = norm_counts ** 2
        score = 1 - square_counts.sum()
        return score
