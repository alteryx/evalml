
from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    categorical_dtypes,
    numeric_and_boolean_dtypes
)


class InvalidTargetDataCheck(DataCheck):
    """Checks if the target data contains missing or invalid values."""

    def __init__(self, problem_type, n_unique=100):
        """Check if the target is invalid for the specified problem type.

        Arguments:
            n_unique (int): Number of unique target values to store when problem type is binary and target
                incorrectly has more than 2 unique values. Non-negative integer. Defaults to 100. If None, stores all unique values.
        """
        self.problem_type = handle_problem_types(problem_type)
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
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, None, None])
            >>> target_check = InvalidTargetDataCheck('binary')
            >>> assert target_check.validate(X, y) == {"errors": [{"message": "2 row(s) (50.0%) of target values are null",\
                                                                   "data_check_name": "InvalidTargetDataCheck",\
                                                                   "level": "error",\
                                                                   "code": "TARGET_HAS_NULL",\
                                                                   "details": {"num_null_rows": 2, "pct_null_rows": 50}}],\
                                                       "warnings": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }
        if y is None:
            raise ValueError("y cannot be None")

        y = _convert_to_woodwork_structure(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        null_rows = y.isnull()
        if null_rows.any():
            num_null_rows = null_rows.sum()
            pct_null_rows = null_rows.mean() * 100
            messages["errors"].append(DataCheckError(message="{} row(s) ({}%) of target values are null".format(num_null_rows, pct_null_rows),
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.TARGET_HAS_NULL,
                                                     details={"num_null_rows": num_null_rows, "pct_null_rows": pct_null_rows}).to_dict())
        valid_target_types = [dtype for dtype in numeric_and_boolean_dtypes + categorical_dtypes]
        if y.dtype.name not in valid_target_types:

            messages["errors"].append(DataCheckError(message="Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_target_types)),
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.TARGET_UNSUPPORTED_TYPE,
                                                     details={"unsupported_type": y.dtype}).to_dict())

        value_counts = y.value_counts()
        unique_values = value_counts.index.tolist()

        if self.problem_type == ProblemTypes.BINARY and len(value_counts) != 2:
            if self.n_unique is None:
                details = {"target_values": unique_values}
            else:
                details = {"target_values": unique_values[:min(self.n_unique, len(unique_values))]}
            messages["errors"].append(DataCheckError(message="Target does not have two unique values which is not supported for binary classification",
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.TARGET_BINARY_NOT_TWO_UNIQUE_VALUES,
                                                     details=details).to_dict())

        if len(value_counts) == 2 and y.dtype in numeric_and_boolean_dtypes:
            if set(unique_values) != set([0, 1]):
                messages["warnings"].append(DataCheckWarning(message="Numerical binary classification target classes must be [0, 1], got [{}] instead".format(", ".join([str(val) for val in unique_values])),
                                                             data_check_name=self.name,
                                                             message_code=DataCheckMessageCode.TARGET_BINARY_INVALID_VALUES,
                                                             details={"target_values": unique_values}).to_dict())

        return messages
