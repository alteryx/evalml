import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError

from evalml.utils.gen_utils import numeric_and_boolean_dtypes


class InvalidTargetDataCheck(DataCheck):
    """Checks if the target labels contain missing or invalid data."""

    def validate(self, X, y):
        """Checks if the target labels contain missing or invalid data.

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list): Features. Ignored.
            y: Target labels to check for invalid data.

        Returns:
            list (DataCheckError): list with DataCheckErrors if any invalid data is found in target labels.

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, None, None])
            >>> target_check = InvalidTargetDataCheck()
            >>> assert target_check.validate(X, y) == [DataCheckError("2 row(s) (50.0%) of target values are null", "InvalidTargetDataCheck")]
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        messages = []
        null_rows = y.isnull()
        if null_rows.any():
            messages.append(DataCheckError("{} row(s) ({}%) of target values are null".format(null_rows.sum(), null_rows.mean() * 100), self.name))
        valid_target_types = numeric_and_boolean_dtypes + ['object', 'category']

        if y.dtype.name not in valid_target_types:
            messages.append(DataCheckError("Target is unsupported {} type. Valid target types include: {}".format(y.dtype, ", ".join(valid_target_types)), self.name))

        value_counts = y.value_counts()
        if len(value_counts) == 2 and y.dtype in numeric_and_boolean_dtypes:
            unique_values = value_counts.index.tolist()
            if set(unique_values) != set([0, 1]):
                messages.append(DataCheckError("Numerical binary classification target classes must be [0, 1], got [{}] instead".format(", ".join([str(val) for val in unique_values])), self.name))

        return messages
