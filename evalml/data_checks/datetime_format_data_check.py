"""Data check that checks if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order to be supported by time series estimators."""
import pandas as pd

from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils import infer_feature_types


class DateTimeFormatDataCheck(DataCheck):
    """Check if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order to be supported by time series estimators.

    Args:
        datetime_column (str, int): The name of the datetime column. If the datetime values are in the index, then pass "index".
    """

    def __init__(self, datetime_column="index"):
        self.datetime_column = datetime_column

    def validate(self, X, y):
        """Checks if the target data has equal intervals and is sorted.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Target data.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if unequal intervals are found in the datetime column.

        Example:
            >>> from pandas as pd
            >>> X = pd.DataFrame(pd.date_range("January 1, 2021", periods=8), columns=["dates"])
            >>> y = pd.Series([1, 2, 4, 2, 1, 2, 3, 1])
            >>> X.iloc[7] = "January 9, 2021"
            >>> datetime_format_check = DateTimeFormatDataCheck()
            >>> assert datetime_format_check.validate(X, y) == {
            ...     "errors": [{"message": "No frequency could be detected in dates, possibly due to uneven intervals.",
            ...                 "data_check_name": "EqualIntervalDataCheck",
            ...                 "level": "error",
            ...                 "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...                 "details": {}}],
            ...     "warnings": [],
            ...     "actions": []}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        y = infer_feature_types(y)

        no_dt_found = False

        if self.datetime_column != "index":
            datetime_values = X[self.datetime_column]
        else:
            datetime_values = X.index
            if not isinstance(datetime_values, pd.DatetimeIndex):
                datetime_values = y.index
            if not isinstance(datetime_values, pd.DatetimeIndex):
                no_dt_found = True

        try:
            inferred_freq = pd.infer_freq(datetime_values)
        except TypeError:
            no_dt_found = True

        if no_dt_found:
            results["errors"].append(
                DataCheckError(
                    message=f"Datetime information could not be found in the data, or was not in a supported datetime format.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_INFORMATION_NOT_FOUND,
                ).to_dict()
            )
            return results

        if not inferred_freq:
            col_name = (
                self.datetime_column
                if self.datetime_column != "index"
                else "either index"
            )
            results["errors"].append(
                DataCheckError(
                    message=f"No frequency could be detected in {col_name}, possibly due to uneven intervals.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                ).to_dict()
            )

        if not (pd.DatetimeIndex(datetime_values).is_monotonic_increasing):
            results["errors"].append(
                DataCheckError(
                    message="Datetime values must be sorted in ascending order.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                ).to_dict()
            )

        return results
