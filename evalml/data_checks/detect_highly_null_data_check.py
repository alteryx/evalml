import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class DetectHighlyNullDataCheck(DataCheck):

    def __init__(self, percent_threshold=0.95):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            percent_threshold(float): If the percentage of values in an input feature exceeds this amount,
                that feature will be considered highly-null. Defaults to 0.95.

        """
        if percent_threshold < 0 or percent_threshold > 1:
            raise ValueError("percent_threshold must be a float between 0 and 1, inclusive.")
        self.percent_threshold = percent_threshold

    def validate(self, X, y=None):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            X (pd.DataFrame, pd.Series, np.array, list) : features
            y : Ignored.

        Returns:
            list (DataCheckWarning): list with a DataCheckWarning if there are any highly-null columns.

        Example:
            >>> df = pd.DataFrame({
            ...    'lots_of_null': [None, None, None, None, 5],
            ...    'no_null': [1, 2, 3, 4, 5]
            ... })
            >>> null_check = DetectHighlyNullDataCheck(percent_threshold=0.8)
            >>> assert null_check.validate(df) == [DataCheckWarning("Columns 'lots_of_null' are more than 80.0% null", "DetectHighlyNullDataCheck")]
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        percent_null = (X.isnull().mean()).to_dict()
        if self.percent_threshold == 0.0:
            has_null_cols = {key: value for key, value in percent_null.items() if value > self.percent_threshold}
            warning_msg = "Column '{}' is more than 0% null"
            return [DataCheckWarning(warning_msg.format(col_name), self.name) for col_name in has_null_cols]
        elif self.percent_threshold == 1.0:
            all_null_cols = {key: value for key, value in percent_null.items() if value == self.percent_threshold}
            warning_msg = "Column '{}' is 100% null"
            return [DataCheckWarning(warning_msg.format(col_name), self.name) for col_name in all_null_cols]
        else:
            highly_null_cols = {key: value for key, value in percent_null.items() if value >= self.percent_threshold}
            warning_msg = "Column '{}' is {}% or more null"
        return [DataCheckWarning(warning_msg.format(col_name, self.percent_threshold * 100), self.name) for col_name in highly_null_cols]
