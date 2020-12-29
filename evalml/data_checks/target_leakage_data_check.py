from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class TargetLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target."""

    def __init__(self, pct_corr_threshold=0.95):
        """Check if any of the features are highly correlated with the target.

        Supports all target and feature types.

        Arguments:
            pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.

        """
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
        self.pct_corr_threshold = pct_corr_threshold

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target.

        Supports all target and feature types.

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
            ...    'y': [13, 5, 13, 74, 24],
            ... })
            >>> y = pd.Series([10, 42, 31, 51, 40])
            >>> target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.95)
            >>> assert target_leakage_check.validate(X, y) == {"warnings": [{"message": "Column 'leak' is 95.0% or more correlated with the target",\
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
        combined = X.copy()
        combined['target'] = y
        combined = _convert_to_woodwork_structure(combined)
        mutual_info = combined.mutual_information()
        if len(mutual_info) == 0:
            return messages
        corr_df = mutual_info[(mutual_info['mutual_info'] >= self.pct_corr_threshold) & ((mutual_info['column_1'] == 'target') | (mutual_info['column_2'] == 'target'))]
        if len(corr_df) == 0:
            return messages

        highly_corr_cols = [row['column_1'] if row['column_1'] != 'target' else row['column_2'] for i, row in corr_df.iterrows()]
        warning_msg = "Column '{}' is {}% or more correlated with the target"
        messages["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name, self.pct_corr_threshold * 100),
                                                      data_check_name=self.name,
                                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                                      details={"column": col_name}).to_dict()
                                     for col_name in highly_corr_cols])
        return messages
