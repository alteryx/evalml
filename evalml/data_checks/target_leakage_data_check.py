"""Data check that checks if any of the features are highly correlated with the target by using mutual information or Pearson correlation."""
from woodwork.config import CONFIG_DEFAULTS

from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils.woodwork_utils import infer_feature_types


class TargetLeakageDataCheck(DataCheck):
    """Check if any of the features are highly correlated with the target by using mutual information, Pearson correlation, and other correlation metrics.

    If method='mutual_info', this data check uses mutual information and supports all target and feature types.
    Other correlation metrics only support binary with numeric and boolean dtypes. This method will return a value in [-1, 1] if other correlation metrics are selected
    and will returns a value in [0, 1] if mutual information is selected. Correlation metrics available can be found in Woodwork's
    `dependence_dict method <https://woodwork.alteryx.com/en/stable/generated/woodwork.table_accessor.WoodworkTableAccessor.dependence_dict.html#woodwork.table_accessor.WoodworkTableAccessor.dependence_dict>`_.

    Args:
        pct_corr_threshold (float): The correlation threshold to be considered leakage. Defaults to 0.95.
        method (string): The method to determine correlation. Use 'max' for the maximum correlation, or for specific correlation metrics, use their name (ie 'mutual_info' for mutual information, 'pearson' for Pearson correlation, etc).
            possible methods can be found in Woodwork's `config <https://woodwork.alteryx.com/en/stable/guides/setting_config_options.html?highlight=config#Viewing-Config-Settings>`_, under `correlation_metrics`.
            Excludes 'all'. Defaults to 'mutual_info'.
    """

    def __init__(self, pct_corr_threshold=0.95, method="mutual_info"):
        if pct_corr_threshold < 0 or pct_corr_threshold > 1:
            raise ValueError(
                "pct_corr_threshold must be a float between 0 and 1, inclusive.",
            )
        methods = CONFIG_DEFAULTS["correlation_metrics"]
        if method not in methods:
            raise ValueError(
                f"Method '{method}' not in available correlation methods. Available methods include {methods}",
            )
        if method == "all":
            raise ValueError("Cannot use 'all' as the method")
        self.pct_corr_threshold = pct_corr_threshold
        self.method = method

    def _calculate_dependence(self, X, y):
        highly_corr_cols = []
        X2 = X.ww.copy()
        target_str = "target_y"
        while target_str in list(X2.columns):
            target_str += "_y"
        X2.ww[target_str] = y
        try:
            dep_corr = X2.ww.dependence_dict(
                measures=self.method,
                target_col=target_str,
            )
        except KeyError:
            # keyError raised when the target does not appear due to incompatibility with the metric, return []
            return []
        highly_corr_cols = sorted(
            [
                corr_info["column_1"]
                for corr_info in dep_corr
                if abs(corr_info[self.method]) >= self.pct_corr_threshold
            ],
            key=lambda x: X2.columns.tolist().index(x),
        )
        return highly_corr_cols

    def validate(self, X, y):
        """Check if any of the features are highly correlated with the target by using mutual information, Pearson correlation, and/or Spearman correlation.

        If `method='mutual_info'` or `'method='max'`, supports all target and feature types. Other correlation metrics only support binary with numeric and boolean dtypes.
        This method will return a value in [-1, 1] if other correlation metrics are selected and will returns a value in [0, 1] if mutual information is selected.

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


            The default method can be changed to pearson from mutual_info.

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
