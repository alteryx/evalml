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

        Examples:
            >>> import pandas as pd
            ...
            >>> class SeriesWrap():
            ...     def __init__(self, series):
            ...         self.series = series
            ...
            ...     def __eq__(self, series_2):
            ...         return all(self.series.eq(series_2.series))
            ...
            >>> df = pd.DataFrame({
            ...     'all_null': [None, pd.NA, None, None, None],
            ...     'lots_of_null': [None, None, None, None, 5],
            ...     'few_null': ["near", "far", pd.NaT, "wherever", "nowhere"],
            ...     'no_null': [1, 2, 3, 4, 5]
            ... })
            ...
            >>> highly_null_dc = HighlyNullDataCheck(pct_null_col_threshold=0.50)
            >>> assert highly_null_dc.validate(df) == {
            ...     'warnings': [{'message': "Columns 'all_null', 'lots_of_null' are 50.0% or more null",
            ...                   'data_check_name': 'HighlyNullDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': ['all_null', 'lots_of_null'],
            ...                               'rows': None,
            ...                               'pct_null_rows': {'all_null': 1.0, 'lots_of_null': 0.8},
            ...                               'null_row_indices': {'all_null': [0, 1, 2, 3, 4],
            ...                                                    'lots_of_null': [0, 1, 2, 3]}},
            ...                   'code': 'HIGHLY_NULL_COLS'}],
            ...     'errors': [],
            ...     'actions': [{'code': 'DROP_COL',
            ...                  'data_check_name': 'HighlyNullDataCheck',
            ...                  'metadata': {'columns': ['all_null', 'lots_of_null'], 'rows': None}}]}
            ...
            ...
            >>> highly_null_dc = HighlyNullDataCheck(pct_null_row_threshold=0.50)
            >>> validation_results = highly_null_dc.validate(df)
            >>> validation_results['warnings'][0]['details']['pct_null_cols'] = SeriesWrap(validation_results['warnings'][0]['details']['pct_null_cols'])
            >>> highly_null_rows = SeriesWrap(pd.Series([0.5, 0.5, 0.75, 0.5]))
            >>> assert validation_results == {
            ...     'warnings': [{'message': '4 out of 5 rows are 50.0% or more null',
            ...                   'data_check_name': 'HighlyNullDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': None,
            ...                               'rows': [0, 1, 2, 3],
            ...                               'pct_null_cols': highly_null_rows},
            ...                   'code': 'HIGHLY_NULL_ROWS'},
            ...                  {'message': "Columns 'all_null' are 95.0% or more null",
            ...                   'data_check_name': 'HighlyNullDataCheck',
            ...                   'level': 'warning',
            ...                   'details': {'columns': ['all_null'],
            ...                               'rows': None,
            ...                               'pct_null_rows': {'all_null': 1.0},
            ...                               'null_row_indices': {'all_null': [0, 1, 2, 3, 4]}},
            ...                   'code': 'HIGHLY_NULL_COLS'}],
            ...     'errors': [],
            ...     'actions': [{'code': 'DROP_ROWS',
            ...                  'data_check_name': 'HighlyNullDataCheck',
            ...                  'metadata': {'columns': None, 'rows': [0, 1, 2, 3]}},
            ...                 {'code': 'DROP_COL',
            ...                  'data_check_name': 'HighlyNullDataCheck',
            ...                  'metadata': {'columns': ['all_null'], 'rows': None}}]}

        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X, ignore_nullable_types=True)

        percent_null_rows = X.isnull().mean(axis=1)
        highly_null_rows = percent_null_rows[
            percent_null_rows >= self.pct_null_row_threshold
        ]
        if len(highly_null_rows) > 0:
            warning_msg = f"{len(highly_null_rows)} out of {len(X)} rows are {self.pct_null_row_threshold*100}% or more null"
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
                    data_check_name=self.name,
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
                    data_check_name=self.name,
                    metadata={"columns": list(highly_null_cols)},
                ).to_dict()
            )
        return results
