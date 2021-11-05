"""Data check that checks each column in the input for datetime features and will issue an error if NaN values are present."""

from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.woodwork_utils import infer_feature_types

error_contains_nan = "Input datetime column(s) ({}) contains NaN values. Please impute NaN values or drop these rows or columns."


class DateTimeNaNDataCheck(DataCheck):
    """Check each column in the input for datetime features and will issue an error if NaN values are present."""

    def validate(self, X, y=None):
        """Check if any datetime columns contain NaN values.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Ignored.  Defaults to None.

        Returns:
            dict: dict with a DataCheckError if NaN values are present in datetime columns.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            ...
            >>> dates = [['2-1-21', '3-1-21'],
            ...         ['2-2-21', '3-2-21'],
            ...         ['2-3-21', '3-3-21'],
            ...         ['2-4-21', '3-4-21']]
            >>> df = pd.DataFrame(dates, columns=['index', "days"])
            >>> dt_nan_dc = DateTimeNaNDataCheck()
            >>> assert dt_nan_dc.validate(df) == {'warnings': [], 'errors': [], 'actions': []}
            ...
            ...
            >>> dates[0][0] = np.datetime64('NaT')
            >>> df = pd.DataFrame(dates, columns=['index', "days"])
            >>> assert dt_nan_dc.validate(df) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop these rows or columns.',
            ...                 'data_check_name': 'DateTimeNaNDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': ['index'], 'rows': None},
            ...                 'code': 'DATETIME_HAS_NAN'}],
            ...     'actions': []}
            ...
            ...
            >>> dates[0][1] = None
            >>> df = pd.DataFrame(dates, columns=['index', "days"])
            >>> assert dt_nan_dc.validate(df) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Input datetime column(s) (index, days) contains NaN values. Please impute NaN values or drop these rows or columns.',
            ...                 'data_check_name': 'DateTimeNaNDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': ['index', 'days'], 'rows': None},
            ...                 'code': 'DATETIME_HAS_NAN'}],
            ...     'actions': []}
            ...
            ...
            >>> dates[0][1] = pd.NA
            >>> df = pd.DataFrame(dates, columns=['index', "days"])
            >>> assert dt_nan_dc.validate(df) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Input datetime column(s) (index, days) contains NaN values. Please impute NaN values or drop these rows or columns.',
            ...                 'data_check_name': 'DateTimeNaNDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': ['index', 'days'], 'rows': None},
            ...                 'code': 'DATETIME_HAS_NAN'}],
            ...     'actions': []}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        datetime_cols = X.ww.select("datetime")
        nan_columns = datetime_cols.columns[datetime_cols.isna().any()].tolist()
        if len(nan_columns) > 0:
            nan_columns = [str(col) for col in nan_columns]
            cols_str = (
                ", ".join(nan_columns) if len(nan_columns) > 1 else nan_columns[0]
            )
            results["errors"].append(
                DataCheckError(
                    message=error_contains_nan.format(cols_str),
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                    details={"columns": nan_columns},
                ).to_dict()
            )
        return results
