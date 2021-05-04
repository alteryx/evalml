from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types


class HighlyNullRowsDataCheck(DataCheck):
    """Checks if there are any highly-null rows in the input."""

    def __init__(self, pct_null_threshold=0.5):
        """Checks if there are any highly-null rows in the input.

        Arguments:
            pct_null_threshold(float): If the percentage of NaN values in an input feature exceeds this amount,
                that feature will be considered highly-null. Defaults to 0.95.

        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        self.pct_null_threshold = pct_null_threshold

    def validate(self, X, y=None):
        """Checks if there are any highly-null rows in the input.

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): Features
            y (ww.DataColumn, pd.Series, np.ndarray): Ignored.

        Returns:
            dict: dict with a DataCheckWarning if there are any highly-null rows.

        Example:
            >>> import pandas as pd
            >>> data = pd.DataFrame({'a': [None, None, 10],
            ...             'b': [None, "text", "text_1"]})
            >>> null_check = HighlyNullRowsDataCheck(pct_null_threshold=0.5)
            >>> assert null_check.validate(data) == {
            ...                'warnings': [DataCheckWarning(message="Row '0' is 50.0% or more null",
            ...                                            data_check_name=HighlyNullRowsDataCheck.name,
            ...                                            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            ...                                            details={'row': 0, 'pct_null_cols': 1.0}).to_dict(),
            ...                            DataCheckWarning(message="Row '1' is 50.0% or more null",
            ...                                            data_check_name=HighlyNullRowsDataCheck.name,
            ...                                            message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
            ...                                            details={'row': 1, 'pct_null_cols': 0.5}).to_dict()],
            ...                'errors': [],
            ...                'actions': [DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 0}).to_dict(),
            ...                            DataCheckAction(DataCheckActionCode.DROP_ROW, metadata={"row": 1}).to_dict()]}
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())

        percent_null = (X.isnull().mean(axis=1)).to_dict()
        highly_null_rows = []
        if self.pct_null_threshold == 0.0:
            highly_null_rows = {key: value for key, value in percent_null.items() if value > 0.0}
            warning_msg = "Row '{}' is more than 0% null"
            results["warnings"].extend([DataCheckWarning(message=warning_msg.format(row_name),
                                                         data_check_name=self.name,
                                                         message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                                         details={"row": row_name, "pct_null_cols": highly_null_rows[row_name]}).to_dict()
                                        for row_name in highly_null_rows])
        else:
            highly_null_rows = {key: value for key, value in percent_null.items() if value >= self.pct_null_threshold}
            warning_msg = "Row '{}' is {}% or more null"
            results["warnings"].extend([DataCheckWarning(message=warning_msg.format(row_name, self.pct_null_threshold * 100),
                                                         data_check_name=self.name,
                                                         message_code=DataCheckMessageCode.HIGHLY_NULL_ROWS,
                                                         details={"row": row_name, "pct_null_cols": highly_null_rows[row_name]}).to_dict()
                                        for row_name in highly_null_rows])

        results["actions"].extend([DataCheckAction(DataCheckActionCode.DROP_ROW,
                                                   metadata={"row": row_name}).to_dict()
                                   for row_name in highly_null_rows])
        return results
