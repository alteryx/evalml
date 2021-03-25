from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)

error_contains_nan = "Input datetime column ({}) contains NaN values. Please input NaN values or drop this column."


class DateTimeNaNDataCheck(DataCheck):
    """Checks if datetime columns contain NaN values."""

    def __init__(self):
        """Checks each column in the input for datetime features and will issue and error if NaN values are present.
        """

    def validate(self, X, y=None):
        """Checks if any datetime columns contain NaN values.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features.
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.  Defaults to None.

        Returns:
            dict: dict with a DataCheckError if NaN values are present in datetime columns.

        Example:
            >>> import pandas as pd
            >>> import woodwork as ww
            >>> import numpy as np
            >>> dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-08'))
            >>> dates[0] = np.datetime64('NaT')
            >>> ww_input = ww.DataTable(pd.DataFrame(dates, columns=['index']))
            >>> dt_nan_check = DateTimeNaNDataCheck()
            >>> assert dt_nan_check.validate(ww_input) == {"warnings": [],
            ...                                             "actions": [],
            ...                                             "errors": [DataCheckError(message='Input datetime column (index) contains NaN values. Please input NaN values or drop this column.',
            ...                                                                     data_check_name=DateTimeNaNDataCheck.name,
            ...                                                                     message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
            ...                                                                     details={"column": 'index'})]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        datetime_cols = _convert_woodwork_types_wrapper(X.select("datetime").to_dataframe())
        if len(datetime_cols) > 0:
            nan_columns = datetime_cols.columns[datetime_cols.isna().any()].tolist()
            results["errors"].extend([DataCheckError(message=error_contains_nan.format(col_name),
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                                                     details={"column": col_name})
                                      for col_name in nan_columns])
        return results
