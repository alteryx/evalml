import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning

from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class NoVarianceDataCheck(DataCheck):

    def __init__(self, count_nan_as_value=False):

        self.dropnan = not count_nan_as_value

    def _check_for_errors(self, column_name, count_unique, any_nulls):

        message = f"{column_name} {int(count_unique)} unique value."

        if count_unique <= 1:

            return DataCheckWarning(message.format(name=column_name), self.name)

        elif count_unique == 2 and not self.dropnan and any_nulls:
            logger.warning(f"{column_name} two unique values including nulls. Consider encoding the nulls for "
                           "this column to be useful for machine learning.")

    def validate(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        unique_counts = X.nunique(dropna=self.dropnan).to_dict()
        any_nulls = (X.isnull().any()).to_dict()

        messages = []

        for name in unique_counts:

            message = self._check_for_errors(f"Column {name} has", unique_counts[name], any_nulls[name])

            if message:

                messages.append(message)

        label_message = self._check_for_errors("The Labels have", y.nunique(dropna=self.dropnan), y.isnull().any())

        if label_message:
            messages.append(label_message)

        return messages
