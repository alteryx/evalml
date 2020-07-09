import pandas as pd

from .data_check import DataCheck
from .data_check_message import DataCheckWarning


class LabelLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target."""

    def __init__(self, pct_corr_threshold=0.95):
        """Check if any of the features are highly correlated with the target.

        Currently only supports binary and numeric targets and features.

        Arguments:
            pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.

        """
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
        self.pct_corr_threshold = pct_corr_threshold

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target.

        Currently only supports binary and numeric targets and features.

        Arguments:
            X (pd.DataFrame): The input features to check
            y (pd.Series): The labels

        Returns:
            list (DataCheckWarning): list with a DataCheckWarning if there is label leakage detected.

        Example:
            >>> X = pd.DataFrame({
            ...    'leak': [10, 42, 31, 51, 61],
            ...    'x': [42, 54, 12, 64, 12],
            ...    'y': [12, 5, 13, 74, 24],
            ... })
            >>> y = pd.Series([10, 42, 31, 51, 40])
            >>> label_leakage_check = LabelLeakageDataCheck(pct_corr_threshold=0.8)
            >>> assert label_leakage_check.validate(X, y) == [DataCheckWarning("Column 'leak' is 80.0% or more correlated with the target", "LabelLeakageDataCheck")]
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']
        if y.dtype not in numerics:
            return []

        X = X.select_dtypes(include=numerics)
        if len(X.columns) == 0:
            return []

        corrs = {label: abs(y.corr(col)) for label, col in X.iteritems() if abs(y.corr(col)) >= self.pct_corr_threshold}
        highly_corr_cols = {key: value for key, value in corrs.items() if value >= self.pct_corr_threshold}
        warning_msg = "Column '{}' is {}% or more correlated with the target"
        return [DataCheckWarning(warning_msg.format(col_name, self.pct_corr_threshold * 100), self.name) for col_name in highly_corr_cols]
