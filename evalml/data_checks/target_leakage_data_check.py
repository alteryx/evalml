"""Data check that checks if any of the features are highly correlated with the target by using mutual information or Pearson correlation."""

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils.woodwork_utils import (
    infer_feature_types,
    numeric_and_boolean_ww,
)


class TargetLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target by using mutual information or Pearson correlation.

    If `method='mutual'`, this data check uses mutual information and supports all target and feature types.
    Otherwise, if `method='pearson'`, it uses Pearson correlation and only supports binary with numeric and boolean dtypes.
    Pearson correlation returns a value in [-1, 1], while mutual information returns a value in [0, 1].

    Args:
        pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.
        method (string): The method to determine correlation. Use 'mutual' for mutual information, otherwise 'pearson' for Pearson correlation. Defaults to 'mutual'.
    """

    def __init__(self, pct_corr_threshold=0.95, method="mutual"):
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError(
                "pct_corr_threshold must be a float between 0 and 1, inclusive."
            )
        if method not in ["mutual", "pearson"]:
            raise ValueError(f"Method '{method}' not in ['mutual', 'pearson']")
        self.pct_corr_threshold = pct_corr_threshold
        self.method = method

    def _calculate_pearson(self, X, y):
        highly_corr_cols = []
        X_num = X.ww.select(include=numeric_and_boolean_ww)
        if (
            y.ww.logical_type.type_string not in numeric_and_boolean_ww
            or len(X_num.columns) == 0
        ):
            return highly_corr_cols
        highly_corr_cols = [
            label
            for label, col in X_num.iteritems()
            if abs(y.corr(col)) >= self.pct_corr_threshold
        ]
        return highly_corr_cols

    def _calculate_mutual_information(self, X, y):
        highly_corr_cols = []
        for col in X.columns:
            cols_to_compare = X.ww[[col]]
            cols_to_compare.ww[str(col) + "y"] = y
            mutual_info = cols_to_compare.ww.mutual_information()
            if (
                len(mutual_info) > 0
                and mutual_info["mutual_info"].iloc[0] > self.pct_corr_threshold
            ):
                highly_corr_cols.append(col)
        return highly_corr_cols

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target by using mutual information or Pearson correlation.

        If `method='mutual'`, supports all target and feature types. Otherwise, if `method='pearson'` only supports binary with numeric and boolean dtypes.
        Pearson correlation returns a value in [-1, 1], while mutual information returns a value in [0, 1].

        Args:
            X (pd.DataFrame, np.ndarray): The input features to check.
            y (pd.Series, np.ndarray): The target data.

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
            >>> assert target_leakage_check.validate(X, y) == {
            ...     "warnings": [{"message": "Column 'leak' is 95.0% or more correlated with the target",
            ...                   "data_check_name": "TargetLeakageDataCheck",
            ...                   "level": "warning",
            ...                   "code": "TARGET_LEAKAGE",
            ...                   "details": {"column": "leak"}}],
            ...     "errors": [],
            ...     "actions": [{"code": "DROP_COL",
            ...                  "metadata": {"column": "leak"}}]}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        y = infer_feature_types(y)

        if self.method == "pearson":
            highly_corr_cols = self._calculate_pearson(X, y)
        else:
            highly_corr_cols = self._calculate_mutual_information(X, y)

        warning_msg = "Column '{}' is {}% or more correlated with the target"
        results["warnings"].extend(
            [
                DataCheckWarning(
                    message=warning_msg.format(col_name, self.pct_corr_threshold * 100),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                    details={"column": col_name},
                ).to_dict()
                for col_name in highly_corr_cols
            ]
        )
        results["actions"].extend(
            [
                DataCheckAction(
                    DataCheckActionCode.DROP_COL, metadata={"column": col_name}
                ).to_dict()
                for col_name in highly_corr_cols
            ]
        )
        return results
