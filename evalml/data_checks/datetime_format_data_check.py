"""Data check that checks if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order to be supported by time series estimators."""
import numpy as np
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

            The column "dates" has a set of dates with hourly frequency appended to the end of a series of days, which is inconsistent
            with the frequency of the previous 9 dates (1 day).

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=6).append(pd.date_range("2021-01-07", periods=3, freq="H")), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "No frequency could be detected in dates, possibly due to uneven intervals.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      }
            ... ]

            The column "dates" has the date 2021-01-31 appended to the end, which implies there are many dates missing.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=9).append(pd.date_range("2021-01-31", periods=1)), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "dates has datetime values missing between start and end date around row(s) [8]",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_IS_MISSING_VALUES",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      }
            ... ]

            The column "dates" has a repeat of the date 2021-01-09 appended to the end, which is considered redundant and will raise an error.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=9).append(pd.date_range("2021-01-09", periods=1)), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "dates has more than one row with the same datetime value",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_REDUNDANT_ROW",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      }
            ... ]

            The column "Weeks" passed integers instead of datetime data, which will raise an error.

            >>> X = pd.DataFrame([1, 2, 3, 4], columns=["Weeks"])
            >>> y = pd.Series([0] * 4)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Datetime information could not be found in the data, or was not in a supported datetime format.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None},
            ...         "code": "DATETIME_INFORMATION_NOT_FOUND",
            ...         "action_options": []
            ...      }
            ... ]

            Converting that same integer data to datetime, however, is valid.

            >>> X = pd.DataFrame(pd.to_datetime([1, 2, 3, 4]), columns=["Weeks"])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == []

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", freq="W", periods=10), columns=["Weeks"])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == []

            While the data passed in is of datetime type, time series requires the datetime information in datetime_column
            to be monotonically increasing (ascending).

            >>> X = X.iloc[::-1]
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Datetime values must be sorted in ascending order.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None},
            ...         "code": "DATETIME_IS_NOT_MONOTONIC",
            ...         "action_options": []
            ...      }
            ... ]
        """
        messages = []

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
            messages.append(
                DataCheckError(
                    message=f"Datetime information could not be found in the data, or was not in a supported datetime format.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_INFORMATION_NOT_FOUND,
                ).to_dict()
            )
            return messages

        is_increasing = pd.DatetimeIndex(datetime_values).is_monotonic_increasing
        if not inferred_freq:
            col_name = (
                self.datetime_column
                if self.datetime_column != "index"
                else "either index"
            )
            frequencies = []
            missing = []
            for i in range(len(datetime_values) - 2):
                freq = pd.infer_freq(datetime_values[i : i + 3])
                if freq is None and i not in missing:
                    missing.append(i + 1)
                frequencies.append(freq)

            # Check for only one row per datetime
            if len(np.unique(datetime_values)) < len(datetime_values):
                messages.append(
                    DataCheckError(
                        message=f"{col_name} has more than one row with the same datetime value",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                    ).to_dict()
                )

            # Check for no date missing in ordered dates
            elif len(set(frequencies)) == 2 and is_increasing:
                messages.append(
                    DataCheckError(
                        message=f"{col_name} has datetime values missing between start and end date around row(s) {missing}",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                    ).to_dict()
                )

            # Otherwise, give a more generic uneven interval error
            else:
                messages.append(
                    DataCheckError(
                        message=f"No frequency could be detected in {col_name}, possibly due to uneven intervals.",
                        data_check_name=self.name,
                        message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                    ).to_dict()
                )

        if not is_increasing:
            messages.append(
                DataCheckError(
                    message="Datetime values must be sorted in ascending order.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                ).to_dict()
            )

        return messages
