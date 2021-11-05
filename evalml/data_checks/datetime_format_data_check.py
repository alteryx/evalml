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

        Examples:
            >>> import pandas as pd
            ...
            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=9).append(pd.date_range("2021-01-31", periods=1)), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == {
            ...     "errors": [{"message": "No frequency could be detected in dates, possibly due to uneven intervals.",
            ...                 "data_check_name": "DateTimeFormatDataCheck",
            ...                 "level": "error",
            ...                 "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...                 "details": {"columns": None, "rows": None}
            ...                 }],
            ...     "warnings": [],
            ...     "actions": []}
            ...
            ...
            >>> X = pd.DataFrame([1, 2, 3, 4], columns=["Weeks"])
            >>> y = pd.Series([0] * 4)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Datetime information could not be found in the data, or was not in a supported datetime format.',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None},
            ...                 'code': 'DATETIME_INFORMATION_NOT_FOUND'}],
            ...     'actions': []}
            ...
            ...
            >>> X = pd.DataFrame(pd.to_datetime([1, 2, 3, 4]), columns=["Weeks"])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == {'warnings': [], 'errors': [], 'actions': []}
            ...
            ...
            >>> X = pd.DataFrame(pd.date_range("2021-01-01", freq='W', periods=10), columns=["Weeks"])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == {'warnings': [], 'errors': [], 'actions': []}
            ...
            ...
            >>> X = X.iloc[::-1]
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == {
            ...     'warnings': [],
            ...     'errors': [{'message': 'Datetime values must be sorted in ascending order.',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'level': 'error',
            ...                 'details': {'columns': None, 'rows': None},
            ...                 'code': 'DATETIME_IS_NOT_MONOTONIC'}],
            ...     'actions': []}
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
