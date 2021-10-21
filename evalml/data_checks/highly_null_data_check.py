"""Data check that checks if there are any highly-null columns and rows in the input."""

from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning,
)
from evalml.utils import infer_feature_types


class HighlyNullDataCheck(DataCheck):
    """Check if there are any highly-null columns and rows in the input.

    Args:
        pct_null_col_threshold(float): If the percentage of NaN values in an input feature exceeds this amount,
            that column will be considered highly-null. Defaults to 0.95.
        pct_null_row_threshold(float): If the percentage of NaN values in an input row exceeds this amount,
            that row will be considered highly-null. Defaults to 0.95.
    """

    def __init__(self, pct_null_col_threshold=0.95, pct_null_row_threshold=0.95):
        if pct_null_col_threshold < 0 or pct_null_col_threshold > 1:
            raise ValueError(
                "pct null column threshold must be a float between 0 and 1, inclusive."
            )
        if pct_null_row_threshold < 0 or pct_null_row_threshold > 1:
            raise ValueError(
                "pct null row threshold must be a float between 0 and 1, inclusive."
            )

        self.pct_null_col_threshold = pct_null_col_threshold
        self.pct_null_row_threshold = pct_null_row_threshold

    def validate(self, X, y=None):
        """Check if there are any highly-null columns or rows in the input.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored. Defaults to None.

        Returns:
            dict: dict with a DataCheckWarning if there are any highly-null columns or rows.

        Example:
            >>> import pandas as pd
            >>> class SeriesWrap():
            ...     def __init__(self, series):
            ...         self.series = series
            ...
            ...     def __eq__(self, series_2):
            ...         return all(self.series.eq(series_2.series))
            ...
            >>> df = pd.DataFrame({
            ...    'lots_of_null': [None, None, None, None, 5],
            ...    'no_null': [1, 2, 3, 4, 5]
            ... })
            >>> null_check = HighlyNullDataCheck(pct_null_col_threshold=0.50, pct_null_row_threshold=0.50)
            >>> validation_results = null_check.validate(df)
            >>> validation_results['warnings'][0]['details']['pct_null_cols'] = SeriesWrap(validation_results['warnings'][0]['details']['pct_null_cols'])
            >>> highly_null_rows = SeriesWrap(pd.Series([0.5, 0.5, 0.5, 0.5]))
            >>> assert validation_results == {
            ...     "errors": [],
            ...     "warnings": [{"message": "4 out of 5 rows are more than 50.0% null",
            ...                   "data_check_name": "HighlyNullDataCheck",
            ...                   "level": "warning",
            ...                   "code": "HIGHLY_NULL_ROWS",
            ...                   "details": {"pct_null_cols": highly_null_rows, "columns": None, "rows": [0, 1, 2, 3]}},
            ...                  {"message": "Columns 'lots_of_null' are 50.0% or more null",
            ...                   "data_check_name": "HighlyNullDataCheck",
            ...                   "level": "warning",
            ...                   "code": "HIGHLY_NULL_COLS",
            ...                   "details": {"columns": ["lots_of_null"], "pct_null_rows": {"lots_of_null": 0.8}, "null_row_indices": {"lots_of_null": [0, 1, 2, 3]}, "rows": None}}],
            ...    "actions": [{"code": "DROP_ROWS", "metadata": {"rows": [0, 1, 2, 3], "columns": None}},
            ...                {"code": "DROP_COL", "metadata": {"columns": ["lots_of_null"], "rows": None}}]}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)

        percent_null_rows = X.isnull().mean(axis=1)
        highly_null_rows = percent_null_rows[
            percent_null_rows >= self.pct_null_row_threshold
        ]
        if len(highly_null_rows) > 0:
            warning_msg = f"{len(highly_null_rows)} out of {len(X)} rows are more than {self.pct_null_row_threshold*100}% null"
            results["warnings"].append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                    details={
                        "rows": highly_null_rows.index.tolist(),
                        "pct_null_cols": highly_null_rows,
                    },
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.DROP_ROWS,
                    metadata={"rows": highly_null_rows.index.tolist()},
                ).to_dict()
            )

        percent_null_cols = (X.isnull().mean()).to_dict()
        highly_null_cols = {
            key: value
            for key, value in percent_null_cols.items()
            if value >= self.pct_null_col_threshold and value != 0
        }
        highly_null_cols_indices = {
            col_: X[col_][X[col_].isnull()].index.tolist() for col_ in highly_null_cols
        }
        warning_msg = "Columns {} are {}% or more null"
        if highly_null_cols:
            results["warnings"].append(
                DataCheckWarning(
                    message=warning_msg.format(
                        (", ").join(
                            ["'{}'".format(str(col)) for col in highly_null_cols]
                        ),
                        self.pct_null_col_threshold * 100,
                    ),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.HIGHLY_NULL_COLS,
                    details={
                        "columns": list(highly_null_cols),
                        "pct_null_rows": highly_null_cols,
                        "null_row_indices": highly_null_cols_indices,
                    },
                ).to_dict()
            )
            results["actions"].append(
                DataCheckAction(
                    DataCheckActionCode.DROP_COL,
                    metadata={"columns": list(highly_null_cols)},
                ).to_dict()
            )
        return results
