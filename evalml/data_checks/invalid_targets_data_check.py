import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError


class InvalidTargetDataCheck(DataCheck):
    """Checks if the target labels contain missing or invalid data."""

    def validate(self, X, y):
        """Checks if the target labels contain missing or invalid data.

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list) : Features. Ignored.
            y : Target labels to check for invalid data.

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
        null_rows = y.isnull()
        if not null_rows.any():
            return []
        return [DataCheckError("{} row(s) ({}%) of target values are null".format(null_rows.sum(), null_rows.mean() * 100), self.name)]
