from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)

error_contains_nan = "Input natural language column(s) ({}) contains NaN values. Please impute NaN values or drop these rows or columns."


class NaturalLanguageNaNDataCheck(DataCheck):
    """Checks if natural language columns contain NaN values."""

    def __init__(self):
        """Checks each column in the input for natural language features and will issue an error if NaN values are present.
        """

    def validate(self, X, y=None):
        """Checks if any natural language columns contain NaN values.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features.
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.  Defaults to None.

        Returns:
            dict: dict with a DataCheckError if NaN values are present in natural language columns.

        Example:
            >>> import pandas as pd
            >>> import woodwork as ww
            >>> import numpy as np
            >>> dates = np.arange(np.datetime64('2017-01-01'), np.datetime64('2017-01-08'))
            >>> dates[0] = np.datetime64('NaT')
            >>> ww_input = ww.DataTable(pd.DataFrame(dates, columns=['index']))
            >>> dt_nan_check = NaturalLanguageNaNDataCheck()
            >>> assert dt_nan_check.validate(ww_input) == {"warnings": [],
            ...                                             "actions": [],
            ...                                             "errors": [DataCheckError(message='Input datetime column(s) (index) contains NaN values. Please impute NaN values or drop these rows or columns.',
            ...                                                                     data_check_name=NaturalLanguageNaNDataCheck.name,
            ...                                                                     message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
            ...                                                                     details={"columns": 'index'}).to_dict()]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        nl_cols = _convert_woodwork_types_wrapper(X.select("natural_language").to_dataframe())
        nan_columns = nl_cols.columns[nl_cols.isna().any()].tolist()
        if len(nan_columns) > 0:
            nan_columns = [str(col) for col in nan_columns]
            cols_str = ', '.join(nan_columns) if len(nan_columns) > 1 else nan_columns[0]
            results["errors"].append(DataCheckError(message=error_contains_nan.format(cols_str),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                                    details={"columns": cols_str}).to_dict())
        return results
