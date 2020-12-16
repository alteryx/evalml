from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class HighlyNullDataCheck(DataCheck):
    """Checks if there are any highly-null columns in the input."""

    def __init__(self, pct_null_threshold=0.95):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            pct_null_threshold(float): If the percentage of NaN values in an input feature exceeds this amount,
                that feature will be considered highly-null. Defaults to 0.95.

        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        self.pct_null_threshold = pct_null_threshold

    def validate(self, X, y=None):
        """Checks if there are any highly-null columns in the input.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.

        Returns:
            dict: dict with a DataCheckWarning if there are any highly-null columns.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...    'lots_of_null': [None, None, None, None, 5],
            ...    'no_null': [1, 2, 3, 4, 5]
            ... })
            >>> null_check = HighlyNullDataCheck(pct_null_threshold=0.8)
            >>> assert null_check.validate(df) == {"errors": [],\
                                                   "warnings": [{"message": "Column 'lots_of_null' is 80.0% or more null",\
                                                                 "data_check_name": "HighlyNullDataCheck",\
                                                                 "level": "warning",\
                                                                 "code": "HIGHLY_NULL",\
                                                                 "details": {"column": "lots_of_null"}}]}
        """
        messages = {
            "warnings": [],
            "errors": []
        }

        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        percent_null = (X.isnull().mean()).to_dict()
        if self.pct_null_threshold == 0.0:
            all_null_cols = {key: value for key, value in percent_null.items() if value > 0.0}
            warning_msg = "Column '{}' is more than 0% null"
            messages["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name),
                                                          data_check_name=self.name,
                                                          message_code=DataCheckMessageCode.HIGHLY_NULL,
                                                          details={"column": col_name}).to_dict()
                                         for col_name in all_null_cols])
        else:
            highly_null_cols = {key: value for key, value in percent_null.items() if value >= self.pct_null_threshold}
            warning_msg = "Column '{}' is {}% or more null"
            messages["warnings"].extend([DataCheckWarning(message=warning_msg.format(col_name, self.pct_null_threshold * 100),
                                                          data_check_name=self.name,
                                                          message_code=DataCheckMessageCode.HIGHLY_NULL,
                                                          details={"column": col_name}).to_dict()
                                         for col_name in highly_null_cols])
        return messages
