from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)

error_contains_nan = "Input datetime column ({}) contains NaN values."


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

        # TODO: Add Example.
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        datetime_cols = _convert_woodwork_types_wrapper(X.select("datetime"))

        if len(datetime_cols) > 0:
            nan_columns = datetime_cols.columns[datetime_cols.isna().any()].tolist()
            results["errors"].extend([DataCheckError(message=error_contains_nan.format(col_name),
                                                     data_check_name=self.name,
                                                     message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                                                     details={"column": col_name.to_dict()})
                                      for col_name in nan_columns])
        return results
