"""Data check that checks if any of the features are highly correlated with the target by using mutual information or Pearson correlation."""
import woodwork as ww

from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils.woodwork_utils import infer_feature_types, numeric_and_boolean_ww

try:
    methods = ww.utils.get_valid_correlation_metrics()
except AttributeError:
    methods = ["mutual_info", "pearson", "max"]


class TargetLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target by using mutual information, Pearson correlation, and other correlation metrics.

    If `method='mutual_info'`, this data check uses mutual information and supports all target and feature types.
    Other correlation metrics only support binary with numeric and boolean dtypes, and return a value in [-1, 1], while mutual information returns a value in [0, 1].
    Correlation metrics available can be found in Woodwork's `dependence_dict method <https://woodwork.alteryx.com/en/stable/generated/woodwork.table_accessor.WoodworkTableAccessor.dependence_dict.html#woodwork.table_accessor.WoodworkTableAccessor.dependence_dict>`_.

    Args:
        pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.
        method (string): The method to determine correlation. Use 'max' for the maximum correlation, or for specific correlation metrics, use their name (ie 'mutual_info' for mutual information, 'pearson' for Pearson correlation, etc).
            Defaults to 'max', which will use all available correlation metrics and find the max value.
    """

    def __init__(self, pct_corr_threshold=0.95, method="max"):
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError(
                "pct_corr_threshold must be a float between 0 and 1, inclusive.",
            )
        if method not in methods:
            raise ValueError(f"Method '{method}' not in {methods}")
        self.pct_corr_threshold = pct_corr_threshold
        self.method = method

    def _calculate_dependence(self, X, y):
        highly_corr_cols = []
        try:
            X2 = X.ww.copy()
            X2.ww["target_y"] = y
            dep_corr = X2.ww.dependence_dict(
                measures=self.method,
                target_col="target_y",
            )
            highly_corr_cols = [
                corr_info["column_1"]
                for corr_info in dep_corr
                if abs(corr_info[self.method]) >= self.pct_corr_threshold
            ]
        except TypeError:
            # no parameter for `target_col` yet
            for col in X.columns:
                cols_to_compare = X.ww[[col]]
                cols_to_compare.ww[str(col) + "y"] = y
                corr_info = cols_to_compare.ww.dependence_dict(measures=self.method)
                if (
                    len(corr_info) > 0
                    and abs(corr_info[0][self.method]) >= self.pct_corr_threshold
                ):
                    highly_corr_cols.append(col)
            return highly_corr_cols

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target by using mutual information, Pearson correlation, and/or Spearman correlation.

        If `method='mutual'` or `'method='max'`, supports all target and feature types. Otherwise, if `method='pearson'` or `method='spearman'`, only supports binary with numeric and boolean dtypes.
        Pearson and Spearman correlation returns a value in [-1, 1], while mutual information returns a value in [0, 1].

        Args:
            X (pd.DataFrame, np.ndarray): The input features to check.
            y (pd.Series, np.ndarray): The target data.

        Returns:
            dict (DataCheckWarning): dict with a DataCheckWarning if target leakage is detected.

        Examples:
            >>> import pandas as pd

            Any columns that are strongly correlated with the target will raise a warning. This could be indicative of
            data leakage.

            >>> X = pd.DataFrame({
            ...    "leak": [10, 42, 31, 51, 61] * 15,
            ...    "x": [42, 54, 12, 64, 12] * 15,
            ...    "y": [13, 5, 13, 74, 24] * 15,
            ... })
            >>> y = pd.Series([10, 42, 31, 51, 40] * 15)
            ...
            >>> target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.95)
            >>> assert target_leakage_check.validate(X, y) == [
            ...     {
            ...         "message": "Column 'leak' is 95.0% or more correlated with the target",
            ...         "data_check_name": "TargetLeakageDataCheck",
            ...         "level": "warning",
            ...         "code": "TARGET_LEAKAGE",
            ...         "details": {"columns": ["leak"], "rows": None},
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                 "data_check_name": "TargetLeakageDataCheck",
            ...                 "parameters": {},
            ...                 "metadata": {"columns": ["leak"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]


            The default method can be changed to pearson from max.

            >>> X["x"] = y / 2
            >>> target_leakage_check = TargetLeakageDataCheck(pct_corr_threshold=0.8, method="pearson")
            >>> assert target_leakage_check.validate(X, y) == [
            ...     {
            ...         "message": "Columns 'leak', 'x' are 80.0% or more correlated with the target",
            ...         "data_check_name": "TargetLeakageDataCheck",
            ...         "level": "warning",
            ...         "details": {"columns": ["leak", "x"], "rows": None},
            ...         "code": "TARGET_LEAKAGE",
            ...         "action_options": [
            ...             {
            ...                 "code": "DROP_COL",
            ...                  "data_check_name": "TargetLeakageDataCheck",
            ...                  "parameters": {},
            ...                  "metadata": {"columns": ["leak", "x"], "rows": None}
            ...             }
            ...         ]
            ...     }
            ... ]

        """
        messages = []

        X = infer_feature_types(X)
        y = infer_feature_types(y)

        highly_corr_cols = self._calculate_dependence(X, y)
        warning_msg_singular = "Column {} is {}% or more correlated with the target"
        warning_msg_plural = "Columns {} are {}% or more correlated with the target"

        if highly_corr_cols:
            if len(highly_corr_cols) == 1:
                warning_msg = warning_msg_singular.format(
                    "'{}'".format(str(highly_corr_cols[0])),
                    self.pct_corr_threshold * 100,
                )
            else:
                warning_msg = warning_msg_plural.format(
                    (", ").join(["'{}'".format(str(col)) for col in highly_corr_cols]),
                    self.pct_corr_threshold * 100,
                )
            messages.append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TARGET_LEAKAGE,
                    details={"columns": highly_corr_cols},
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.DROP_COL,
                            data_check_name=self.name,
                            metadata={"columns": highly_corr_cols},
                        ),
                    ],
                ).to_dict(),
            )
        return messages
