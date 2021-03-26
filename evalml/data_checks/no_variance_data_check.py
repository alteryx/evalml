from evalml.data_checks import (
    DataCheck,
    DataCheckAction,
    DataCheckActionCode,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning
)
from evalml.utils import _convert_woodwork_types_wrapper, infer_feature_types
from evalml.utils.logger import get_logger

logger = get_logger(__file__)


class NoVarianceDataCheck(DataCheck):
    """Check if the target or any of the features have no variance."""

    def __init__(self, count_nan_as_value=False):
        """Check if the target or any of the features have no variance.

        Arguments:
            count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
                If set to True, a feature that has one unique value and all other data is missing, a
                DataCheckWarning will be returned instead of an error. Defaults to False.
        """
        self._dropnan = not count_nan_as_value

    def _check_for_errors(self, column_name, count_unique, any_nulls):
        """Checks if a column has no variance.

        Arguments:
            column_name (str): Name of the column we are checking.
            count_unique (float): Number of unique values in this column.
            any_nulls (bool): Whether this column has any missing data.

        Returns:
            DataCheckError if the column has no variance or DataCheckWarning if the column has two unique values including NaN.
        """
        message = f"{column_name} has {int(count_unique)} unique value."

        if count_unique <= 1:
            return DataCheckError(message=message.format(name=column_name),
                                  data_check_name=self.name,
                                  message_code=DataCheckMessageCode.NO_VARIANCE,
                                  details={"column": column_name})

        elif count_unique == 2 and not self._dropnan and any_nulls:
            return DataCheckWarning(message=f"{column_name} has two unique values including nulls. "
                                    "Consider encoding the nulls for "
                                    "this column to be useful for machine learning.",
                                    data_check_name=self.name,
                                    message_code=DataCheckMessageCode.NO_VARIANCE_WITH_NULL,
                                    details={"column": column_name})

    def validate(self, X, y):
        """Check if the target or any of the features have no variance (1 unique value).

        Arguments:
            X (ww.DataTable, pd.DataFrame, np.ndarray): The input features.
            y (ww.DataColumn, pd.Series, np.ndarray): The target data.

        Returns:
            dict: dict of warnings/errors corresponding to features or target with no variance.
        """
        results = {
            "warnings": [],
            "errors": [],
            "actions": []
        }

        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y.to_series())

        unique_counts = X.nunique(dropna=self._dropnan).to_dict()
        any_nulls = (X.isnull().any()).to_dict()
        for col_name in unique_counts:
            message = self._check_for_errors(col_name, unique_counts[col_name], any_nulls[col_name])
            if not message:
                continue
            DataCheck._add_message(message, results)
            results["actions"].append(DataCheckAction(DataCheckActionCode.DROP_COL,
                                                      metadata={"column": col_name}).to_dict())
        y_name = getattr(y, "name")
        if not y_name:
            y_name = "Y"
        target_message = self._check_for_errors(y_name, y.nunique(dropna=self._dropnan), y.isnull().any())
        if target_message:
            DataCheck._add_message(target_message, results)
        return results
