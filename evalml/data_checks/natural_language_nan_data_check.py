from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.woodwork_utils import infer_feature_types

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
            >>> data = pd.DataFrame()
            >>> data['A'] = [None, "string_that_is_long_enough_for_natural_language"]
            >>> data['B'] = ['string_that_is_long_enough_for_natural_language', 'string_that_is_long_enough_for_natural_language']
            >>> data['C'] = np.random.randint(0, 3, size=len(data))
            >>> data = ww.DataTable(data, logical_types={'A': 'NaturalLanguage', 'B': 'NaturalLanguage'})
            >>> nl_nan_check = NaturalLanguageNaNDataCheck()
            >>> assert nl_nan_check.validate(data) == {
            ...        "warnings": [],
            ...        "actions": [],
            ...        "errors": [DataCheckError(message='Input natural language column(s) (A) contains NaN values. Please impute NaN values or drop these rows or columns.',
            ...                      data_check_name=NaturalLanguageNaNDataCheck.name,
            ...                      message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
            ...                      details={"columns": 'A'}).to_dict()]
            ...    }
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        X = X.select('natural_language')
        X_describe = X.describe_dict()
        nan_columns = [str(col) for col in X_describe if X_describe[col]['nan_count'] > 0]
        if len(nan_columns) > 0:
            cols_str = ', '.join(nan_columns)
            results["errors"].append(DataCheckError(message=error_contains_nan.format(cols_str),
                                                    data_check_name=self.name,
                                                    message_code=DataCheckMessageCode.NATURAL_LANGUAGE_HAS_NAN,
                                                    details={"columns": cols_str}).to_dict())
        return results
