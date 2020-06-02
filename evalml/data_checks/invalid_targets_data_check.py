import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError


class InvalidTargetDataCheck(DataCheck):

    def validate(self, X, y):
        """Checks if the target labels contain invalid data.

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list) : Features. Ignored.
            y : Target labels to check for invalid data.

        Returns:
            list (DataCheckError): list with DataCheckErrors if any invalid data is found in target labels.

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, None])
            >>> target_check = InvalidTargetDataCheck()
            >>> assert target_check.validate(X, y) == [DataCheckError("Row(s) 2 contains a null value", "InvalidTargetDataCheck")]
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        null_rows = y.isnull()
        if null_rows.any():
            return [DataCheckError("Row(s) {} contains a null value".format(', '.join([str(row_index) for row_index, row_value in null_rows.items() if row_value])), self.name)]
        return []
