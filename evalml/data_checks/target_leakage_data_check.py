from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    numeric_and_boolean_ww
)


class TargetLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target by using mutual information or Pearson correlation."""

    def __init__(self, pct_corr_threshold=0.95, pearson_corr=False):
        """Check if any of the features are highly correlated with the target by using mutual information or Pearson correlation.

        If `pearson_corr=False`, this data check uses mutual information and supports all target and feature types.
        Otherwise, it uses Pearson correlation and only supports binary with numeric and boolean dtypes.
        Pearson correlation returns a value in [-1, 1], while mutual information returns a value in [0, 1].

        Arguments:
            pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.
            pearson_corr (bool): Whether or not to use the Pearson correlation versus mutual information. Defaults to False, which uses mutual information.

        """
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")
        self.pct_corr_threshold = pct_corr_threshold
        self.pearson = pearson_corr

    def _calculate_pearson(self, X, y):
        highly_corr_cols = []
        X_num = X.select(include=numeric_and_boolean_ww)
        if len(X_num.columns) > 0:
            highly_corr_cols = [label for label, col in X_num.iteritems() if abs(y.corr(col)) >= self.pct_corr_threshold]
        return highly_corr_cols

    def _calculate_mutual_information(self, X, y):
        highly_corr_cols = []
        # safely add in the target column without overlapping with any existing column names
        target = 'target'
        while target in X.columns:
            target += '0'

        combined = X.copy()
        combined[target] = y
        combined = _convert_to_woodwork_structure(combined)
        for col in X.columns:
            sample = combined[[col, target]]
            mutual_info = sample.mutual_information()
            if len(mutual_info) > 0 and mutual_info['mutual_info'].iloc[0] > self.pct_corr_threshold:
                highly_corr_cols.append(col)
        return highly_corr_cols

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target by using mutual information.

        If `pearson_corr=False`, supports all target and feature types. Otherwise, only supports binary with numeric and boolean dtypes.
        Pearson correlation returns a value in [-1, 1], while mutual information returns a value in [0, 1].

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
        if y.logical_type not in numeric_and_boolean_ww and self.pearson:
            return messages
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_series())
        if not self.pearson:
            highly_corr_cols = self._calculate_mutual_information(X, y)
        else:
            highly_corr_cols = self._calculate_pearson(X, y, messages)

        warning_msg = "Column '{}' is {}% or more correlated with the target"
        messages["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name, self.pct_corr_threshold * 100),
                                                      data_check_name=self.name,
                                                      message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                                                      details={"column": col_name}).to_dict()
                                     for col_name in highly_corr_cols])
        return messages
