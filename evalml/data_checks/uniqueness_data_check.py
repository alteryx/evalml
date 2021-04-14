from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.problem_types import (
    handle_problem_types,
    is_multiclass,
    is_regression
)
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)

warning_not_unique_enough = "Input columns ({}) for {} problem type are not unique enough."
warning_too_unique = "Input columns ({}) for {} problem type are too unique."


class UniquenessDataCheck(DataCheck):
    """Checks if there are any columns in the input that are either too unique for classification problems
    or not unique enough for regression problems."""

    def __init__(self, problem_type, threshold=0.50):
        """Checks each column in the input to determine the uniqueness of the values in those columns.

        Arguments:
            problem_type (str or ProblemTypes): The specific problem type to data check for.
                e.g. 'binary', 'multiclass', 'regression, 'time series regression'
            threshold(float): The threshold to set as an upper bound on uniqueness for classification type problems
                or lower bound on for regression type problems.  Defaults to 0.50.

        """
        self.problem_type = handle_problem_types(problem_type)
        if threshold < 0 or threshold > 1:
            raise ValueError("threshold must be a float between 0 and 1, inclusive.")
        self.threshold = threshold

    def validate(self, X, y=None):
        """Checks if there are any columns in the input that are too unique in the case of classification
        problems or not unique enough in the case of regression problems.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features.
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.  Defaults to None.

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
            >>> assert uniqueness_check.validate(df) == {"errors": [],\
                                                         "warnings": [{"message": "Input columns (regression_not_unique_enough) for regression problem type are not unique enough.",\
                                                                 "data_check_name": "UniquenessDataCheck",\
                                                                 "level": "warning",\
                                                                 "code": "NOT_UNIQUE_ENOUGH",\
                                                                 "details": {"column": "regression_not_unique_enough", 'uniqueness_score': 0.0}}],\
                                                         "actions": [{"code": "DROP_COL",\
                                                                      "metadata": {"column": "regression_not_unique_enough"}}]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        res = X.apply(UniquenessDataCheck.uniqueness_score)

        if is_regression(self.problem_type):
            not_unique_enough_cols = list(res.index[res < self.threshold])
            results["warnings"].extend([DataCheckWarning(message=warning_not_unique_enough.format(col_name,
                                                                                                  self.problem_type),
                                                         data_check_name=self.name,
                                                         message_code=DataCheckMessageCode.NOT_UNIQUE_ENOUGH,
                                                         details={"column": col_name, "uniqueness_score": res.loc[col_name]}).to_dict()
                                        for col_name in not_unique_enough_cols])
            results["actions"].extend([DataCheckAction(action_code=DataCheckActionCode.DROP_COL,
                                                       metadata={"column": col_name}).to_dict()
                                       for col_name in not_unique_enough_cols])
        elif is_multiclass(self.problem_type):
            too_unique_cols = list(res.index[res > self.threshold])
            results["warnings"].extend([DataCheckWarning(message=warning_too_unique.format(col_name,
                                                                                           self.problem_type),
                                                         data_check_name=self.name,
                                                         message_code=DataCheckMessageCode.TOO_UNIQUE,
                                                         details={"column": col_name, "uniqueness_score": res.loc[col_name]}).to_dict()
                                        for col_name in too_unique_cols])
            results["actions"].extend([DataCheckAction(action_code=DataCheckActionCode.DROP_COL,
                                                       metadata={"column": col_name}).to_dict()
                                       for col_name in too_unique_cols])
        return results

    @staticmethod
    def uniqueness_score(col):
        """This function calculates a uniqueness score for the provided field.  NaN values are
        not considered as unique values in the calculation.

        Based on the Herfindahl–Hirschman Index.

        Arguments:
            col (pd.Series): Feature values.
        Returns:
            (float): Uniqueness score.
        """
        norm_counts = col.value_counts() / col.value_counts().sum()
        square_counts = norm_counts ** 2
        score = 1 - square_counts.sum()
        return score
