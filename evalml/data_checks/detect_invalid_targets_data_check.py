import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError


class DetectInvalidTargetsDataCheck(DataCheck):

    def validate(self, X, y):
        """Checks if the target labels contain invalid data.

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list) : Features. Ignored.
            y : Target labels to check for invalid data.

        Returns:
            list (DataCheckError): list with DataCheckErrors if any invalid data is found in target labels.
                - abc
                - def

        Example:
            >>> X = pd.DataFrame({})
            >>> y = pd.Series([0, 1, np.nan])
            >>> target_check = DetectInvalidTargetsDataCheck()
            >>> assert target_check.validate(X, y) == [DataCheckError("Row '2' contains a null value", "DetectInvalidTargetsDataCheck")]
        """
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        null_rows = y.isnull()
        error_msg = "Row '{}' contains a null value"
        return [DataCheckError(error_msg.format(row_index), self.name) for row_index, row_value in null_rows.items() if row_value]
