import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class DetectHighlyNullDataCheck(DataCheck):

    def __init__(self, pct_null_threshold=0.95):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            pct_null_threshold(float): If the percentage of values in an input feature exceeds this amount,
                that feature will be considered highly-null. Defaults to 0.95.

        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        self.pct_null_threshold = pct_null_threshold

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
            >>> null_check = DetectHighlyNullDataCheck(pct_null_threshold=0.8)
            >>> assert null_check.validate(df) == [DataCheckWarning("Columns 'lots_of_null' is 80.0% or more null", "DetectHighlyNullDataCheck")]
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        percent_null = (X.isnull().mean()).to_dict()
        if self.pct_null_threshold == 0.0:
            all_null_cols = {key: value for key, value in percent_null.items() if value > 0.0}
            warning_msg = "Column '{}' is more than 0% null"
            return [DataCheckWarning(warning_msg.format(col_name), self.name) for col_name in all_null_cols]
        else:
            highly_null_cols = {key: value for key, value in percent_null.items() if value >= self.pct_null_threshold}
            warning_msg = "Column '{}' is {}% or more null"

            return [DataCheckWarning(warning_msg.format(col_name, self.pct_null_threshold * 100), self.name) for col_name in highly_null_cols]
