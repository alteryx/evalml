"""Data check that checks if there are any columns in the input that are either too unique for classification problems or not unique enough for regression problems."""
from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.problem_types import handle_problem_types, is_multiclass, is_regression
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

        Examples:
            >>> import pandas as pd

            Because the problem type is regression, the column "regression_not_unique_enough" raises a warning
            for having just one value.

            >>> df = pd.DataFrame({
            ...    "regression_unique_enough": [float(x) for x in range(100)],
            ...    "regression_not_unique_enough": [float(1) for x in range(100)]
            ... })
            ...
            >>> uniqueness_check = UniquenessDataCheck(problem_type="regression", threshold=0.8)
            >>> assert uniqueness_check.validate(df) == [
            ...     {
            ...         "message": "Input columns 'regression_not_unique_enough' for regression problem type are not unique enough.",
            ...         "data_check_name": "UniquenessDataCheck",
            ...         "level": "warning",
            ...         "code": "NOT_UNIQUE_ENOUGH",
            ...         "details": {"columns": ["regression_not_unique_enough"], "uniqueness_score": {"regression_not_unique_enough": 0.0}, "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "parameters": {},
            ...                 "data_check_name": "UniquenessDataCheck",
            ...                 "metadata": {"columns": ["regression_not_unique_enough"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]

            For multiclass, the column "regression_unique_enough" has too many unique values and will raise
            an appropriate warning.
            >>> y = pd.Series([1, 1, 1, 2, 2, 3, 3, 3])
            >>> uniqueness_check = UniquenessDataCheck(problem_type="multiclass", threshold=0.8)
            >>> assert uniqueness_check.validate(df) == [
            ...     {
            ...         "message": "Input columns 'regression_unique_enough' for multiclass problem type are too unique.",
            ...         "data_check_name": "UniquenessDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": ["regression_unique_enough"],
            ...             "rows": None,
            ...             "uniqueness_score": {"regression_unique_enough": 0.99}
            ...         },
            ...         "code": "TOO_UNIQUE",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "UniquenessDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["regression_unique_enough"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]
            ...
            >>> assert UniquenessDataCheck.uniqueness_score(y) == 0.65625
        """
        messages = []

        X = infer_feature_types(X)

        res = X.apply(UniquenessDataCheck.uniqueness_score)

        if is_regression(self.problem_type):
            not_unique_enough_cols = list(res.index[res < self.threshold])
            messages.append(
                DataCheckWarning(
                    message=warning_not_unique_enough.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in not_unique_enough_cols],
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
                    action_options=[
                        DataCheckActionOption(
                            action_code=DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": not_unique_enough_cols},
                        ),
                    ],
                ).to_dict(),
            )

        elif is_multiclass(self.problem_type):
            too_unique_cols = list(res.index[res > self.threshold])
            messages.append(
                DataCheckWarning(
                    message=warning_too_unique.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in too_unique_cols],
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
                    action_options=[
                        DataCheckActionOption(
                            action_code=DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": too_unique_cols},
                        ),
                    ],
                ).to_dict(),
            )

        return messages

    @staticmethod
    def uniqueness_score(col, drop_na=True):
        """Calculate a uniqueness score for the provided field.  NaN values are not considered as unique values in the calculation.

        Based on the Herfindahl-Hirschman Index.

        Args:
            col (pd.Series): Feature values.
            drop_na (bool): Whether to drop null values when computing the uniqueness score. Defaults to True.

        Returns:
            (float): Uniqueness score.
        """
        norm_counts = (
            col.value_counts(dropna=drop_na) / col.value_counts(dropna=drop_na).sum()
        )
        square_counts = norm_counts**2
        score = 1 - square_counts.sum()
        return score
