from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    numeric_and_boolean_dtypes
)


class TargetLeakageDataCheck(DataCheck):
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
            X (ww.DataTable, pd.DataFrame, np.ndarray): The input features to check
            y (ww.DataColumn, pd.Series, np.ndarray): The target data

        Returns:
            dict (DataCheckWarning): dict with a DataCheckWarning if target leakage is detected.

        Example:
            >>> import pandas as pd
            >>> X = pd.DataFrame({
            ...    'leak': [10, 42, 31, 51, 61],
            ...    'x': [42, 54, 12, 64, 12],
            ...    'y': [12, 5, 13, 74, 24],
            ... })
            >>> y = pd.Series([10, 42, 31, 51, 40])
            >>> target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8)
            >>> assert target_leakage_check.validate(X, y) == {"warnings": [{"message": "Column 'leak' is 80.0% or more correlated with the target",\
                                                                             "data_check_name": "TargetLeakageDataCheck",\
                                                                             "level": "warning",\
                                                                             "code": "TARGET_LEAKAGE",\
                                                                             "details": {"column": "leak"}}],\
                                                               "errors": []}
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        X = _convert_to_woodwork_structure(X)
        y = _convert_to_woodwork_structure(y)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())

        if y.dtype not in numeric_and_boolean_dtypes:
            return messages
        X = X.select_dtypes(include=numeric_and_boolean_dtypes)
        if len(X.columns) == 0:
            return messages

        highly_corr_cols = {label: abs(y.corr(col)) for label, col in X.iteritems() if abs(y.corr(col)) >= self.pct_corr_threshold}
        warning_msg = "Column '{}' is {}% or more correlated with the target"
        messages["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name, self.pct_corr_threshold * 100),
                                                      data_check_name=self.name,
                                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                                      details={"column": col_name}).to_dict()
                                     for col_name in highly_corr_cols])
        return messages
