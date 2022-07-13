"""Data check that checks if there are any columns with sparsely populated values in the input."""
from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.problem_types import handle_problem_types, is_multiclass
from evalml.utils.woodwork_utils import infer_feature_types

warning_too_unique = "Input columns ({}) for {} problem type are too sparse."


class SparsityDataCheck(DataCheck):
    """Check if there are any columns with sparsely populated values in the input.

    Args:
        problem_type (str or ProblemTypes): The specific problem type to data check for.
            'multiclass' or 'time series multiclass' is the only accepted problem type.
        threshold (float): The threshold value, or percentage of each column's unique values,
            below which, a column exhibits sparsity.  Should be between 0 and 1.
        unique_count_threshold (int): The minimum number of times a unique
            value has to be present in a column to not be considered "sparse."
            Defaults to 10.
    """

    def __init__(self, problem_type, threshold, unique_count_threshold=10):
        self.problem_type = handle_problem_types(problem_type)
        if not is_multiclass(self.problem_type):
            raise ValueError("Sparsity is only defined for multiclass problem types.")
        self.threshold = threshold
        if threshold < 0 or threshold > 1:
            raise ValueError("Threshold must be a float between 0 and 1, inclusive.")
        self.unique_count_threshold = unique_count_threshold
        if unique_count_threshold < 0 or not isinstance(unique_count_threshold, int):
            raise ValueError("Unique count threshold must be positive integer.")

    def validate(self, X, y=None):
        """Calculate what percentage of each column's unique values exceed the count threshold and compare that percentage to the sparsity threshold stored in the class instance.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored.

        Returns:
            dict: dict with a DataCheckWarning if there are any sparse columns.

        Examples:
            >>> import pandas as pd

            For multiclass problems, if a column doesn't have enough representation from unique values, it will be considered sparse.

            >>> df = pd.DataFrame({
            ...    "sparse": [float(x) for x in range(100)],
            ...    "not_sparse": [float(1) for x in range(100)]
            ... })
            ...
            >>> sparsity_check = SparsityDataCheck(problem_type="multiclass", threshold=0.5, unique_count_threshold=10)
            >>> assert sparsity_check.validate(df) == [
            ...     {
            ...         "message": "Input columns ('sparse') for multiclass problem type are too sparse.",
            ...         "data_check_name": "SparsityDataCheck",
            ...         "level": "warning",
            ...         "code": "TOO_SPARSE",
            ...         "details": {
            ...             "columns": ["sparse"],
            ...             "sparsity_score": {"sparse": 0.0},
            ...             "rows": None
            ...         },
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                  "data_check_name": "SparsityDataCheck",
            ...                  "parameters": {},
            ...                  "metadata": {"columns": ["sparse"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]

            ...
            >>> df["sparse"] = [float(x % 10) for x in range(100)]
            >>> sparsity_check = SparsityDataCheck(problem_type="multiclass", threshold=1, unique_count_threshold=5)
            >>> assert sparsity_check.validate(df) == []
            ...
            >>> sparse_array = pd.Series([1, 1, 1, 2, 2, 3] * 3)
            >>> assert SparsityDataCheck.sparsity_score(sparse_array, count_threshold=5) == 0.6666666666666666
        """
        messages = []

        X = infer_feature_types(X)

        res = X.apply(
            SparsityDataCheck.sparsity_score,
            count_threshold=self.unique_count_threshold,
        )
        too_sparse_cols = [col for col in res.index[res < self.threshold]]
        if too_sparse_cols:
            messages.append(
                DataCheckWarning(
                    message=warning_too_unique.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in too_sparse_cols],
                        ),
                        self.problem_type,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TOO_SPARSE,
                    details={
                        "columns": too_sparse_cols,
                        "sparsity_score": {
                            col: res.loc[col] for col in too_sparse_cols
                        },
                    },
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": too_sparse_cols},
                        ),
                    ],
                ).to_dict(),
            )

        return messages

    @staticmethod
    def sparsity_score(col, count_threshold=10):
        """Calculate a sparsity score for the given value counts by calculating the percentage of unique values that exceed the count_threshold.

        Args:
            col (pd.Series): Feature values.
            count_threshold (int): The number of instances below which a value is considered sparse.
                Default is 10.

        Returns:
            (float): Sparsity score, or the percentage of the unique values that exceed count_threshold.
        """
        counts = col.value_counts()
        score = sum(counts > count_threshold) / counts.size

        return score
