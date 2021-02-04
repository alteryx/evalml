from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)

warning_not_unique_enough = "Input columns ({}) for {} problem type are not unique enough."
warning_too_unique = "Input columns ({}) for {} problem type are too unique."


class UniquenessDataCheck(DataCheck):
    """Checks if there are any highly-null columns in the input."""

    def __init__(self, problem_type, threshold=0.50):
        """Checks each column in the input to determine the uniqueness of the values in those columns.

        Arguments:
            problem_type (str or ProblemTypes): The specific problem type to data check for.
                e.g. 'binary', 'multiclass', 'regression, 'time series regression'
            threshold(float): Defaults to 0.50.

        """
        self.problem_type = handle_problem_types(problem_type)
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.

        Returns:
            dict: dict with a DataCheckWarning if there are any highly-null columns.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...    'regression_unique_enough': [float(x) for x in range(100)],
            ...    'regression_not_unique_enough': [float(1) for x in range(100)]
            ... })
            >>> uniqueness_check = UniquenessDataCheck(threshold=0.8)
            >>> assert uniqueness_check.validate(df) == {"errors": [],\
                                                   "warnings": [{"message": "Input columns (regression_not_unique_enough) for regression problem type are not unique enough.",\
                                                                 "data_check_name": "UniquenessDataCheck",\
                                                                 "level": "warning",\
                                                                 "code": "NOT_UNIQUE_ENOUGH",\
                                                                 "details": {"column": "regression_not_unique_enough"}}]}
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        def uniqueness_score(col):
            norm_counts = col.value_counts() / col.value_counts().sum()
            square_counts = norm_counts ** 2
            score = 1 - square_counts.sum()
            return score

        res = X.apply(uniqueness_score)

        if self.problem_type == ProblemTypes.REGRESSION:
            not_unique_enough_cols = list(res.index[res < self.threshold])
            messages["warnings"].extend([DataCheckWarning(message=warning_not_unique_enough.format(col_name,
                                                                                                   self.problem_type),
                                                          data_check_name=self.name,
                                                          message_code=DataCheckMessageCode.NOT_UNIQUE_ENOUGH,
                                                          details={"column": col_name}).to_dict()
                                         for col_name in not_unique_enough_cols])
        elif self.problem_type == ProblemTypes.MULTICLASS:
            too_unique_cols = list(res.index[res > self.threshold])
            messages["warnings"].extend([DataCheckWarning(message=warning_too_unique.format(col_name,
                                                                                            self.problem_type),
                                                          data_check_name=self.name,
                                                          message_code=DataCheckMessageCode.TOO_UNIQUE,
                                                          details={"column": col_name}).to_dict()
                                         for col_name in too_unique_cols])
        return messages
