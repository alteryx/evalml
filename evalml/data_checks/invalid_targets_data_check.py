import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError

from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils.gen_utils import (
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


class InvalidTargetDataCheck(DataCheck):
    """Checks if the target data contains missing or invalid values."""

    def __init__(self, problem_type):
        self.problem_type = handle_problem_types(problem_type)

    def validate(self, X, y):
        """Checks if the target data contains missing or invalid values.

        Arguments:
            X (pd.DataFrame, pd.Series, np.ndarray, list): Features. Ignored.
            y: Target data to check for invalid values.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if any invalid values are found in the target data.

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, None, None])
            >>> target_check = InvalidTargetDataCheck('binary')
            >>> assert target_check.validate(X, y) == {"errors": [{"message": "2 row(s) (50.0%) of target values are null", "data_check_name": "InvalidTargetDataCheck", "level": "error"}],\
                                                       "warnings": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        null_rows = y.isnull()
        if null_rows.any():
            messages["errors"].append(DataCheckError("{} row(s) ({}%) of target values are null".format(null_rows.sum(), null_rows.mean() * 100), self.name).to_dict())
        valid_target_types = [dtype for dtype in numeric_and_boolean_dtypes + categorical_dtypes]
        if y.dtype.name not in valid_target_types:
            messages["errors"].append(DataCheckError("Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_target_types)), self.name).to_dict())

        value_counts = y.value_counts()

        if self.problem_type == ProblemTypes.BINARY and len(value_counts) != 2:
            messages["errors"].append(DataCheckError("Target does not have two unique values which is not supported for binary classification", self.name).to_dict())

        if len(value_counts) == 2 and y.dtype in numeric_and_boolean_dtypes:
            unique_values = value_counts.index.tolist()
            if set(unique_values) != set([0, 1]):
                messages["errors"].append(DataCheckError("Numerical binary classification target classes must be [0, 1], got [{}] instead".format(", ".join([str(val) for val in unique_values])), self.name).to_dict())

        return messages
