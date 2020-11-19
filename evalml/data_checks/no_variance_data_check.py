import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckError, DataCheckWarning
from .data_check_message_type import DataCheckMessageType

from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class NoVarianceDataCheck(DataCheck):
    """Check if the target or any of the features have no variance."""

    def __init__(self, count_nan_as_value=False):
        """Check if the target or any of the features have no variance.

        Arguments:
            count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
                If set to True, a feature that has one unique value and all other data is missing, a
                DataCheckWarning will be returned instead of an error. Defaults to False.
        """
        self._dropnan = not count_nan_as_value

    def _check_for_errors(self, column_name, count_unique, any_nulls):
        """Checks if a column has no variance.

        Arguments:
            column_name (str): Name of the column we are checking.
            count_unique (float): Number of unique values in this column.
            any_nulls (bool): Whether this column has any missing data.

        Returns:
            DataCheckError if the column has no variance or DataCheckWarning if the column has two unique values including NaN.
        """
        message = f"{column_name} has {int(count_unique)} unique value."

        if count_unique <= 1:
            return DataCheckError(message.format(name=column_name), self.name)

        elif count_unique == 2 and not self._dropnan and any_nulls:
            return DataCheckWarning(f"{column_name} has two unique values including nulls. "
                                    "Consider encoding the nulls for "
                                    "this column to be useful for machine learning.", self.name)

    def validate(self, X, y):
        """Check if the target or any of the features have no variance (1 unique value).

        Arguments:
            X (pd.DataFrame): The input features.
            y (pd.Series): The target data.

        Returns:
            dict (DataCheckWarning or DataCheckError): dict of warnings/errors corresponding to features or target with no variance.
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        unique_counts = X.nunique(dropna=self._dropnan).to_dict()
        any_nulls = (X.isnull().any()).to_dict()
        for name in unique_counts:
            message = self._check_for_errors(name, unique_counts[name], any_nulls[name])
            if message:
                if message.message_type == DataCheckMessageType.ERROR:
                    messages["errors"].append(message)
                elif message.message_type == DataCheckMessageType.WARNING:
                    messages["warnings"].append(message)
        y_name = getattr(y, "name")
        if not y_name:
            y_name = "Y"
        target_message = self._check_for_errors(y_name, y.nunique(dropna=self._dropnan), y.isnull().any())
        if target_message:
            if target_message.message_type == DataCheckMessageType.ERROR:
                messages["errors"].append(target_message)
            elif target_message.message_type == DataCheckMessageType.WARNING:
                messages["warnings"].append(target_message)
        return messages
