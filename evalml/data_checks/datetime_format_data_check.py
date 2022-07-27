"""Data check that checks if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order to be supported by time series estimators."""
import pandas as pd
from woodwork.statistics_utils import infer_frequency

from evalml.data_checks import (
    DataCheck,
    DataCheckActionCode,
    DataCheckActionOption,
    DataCheckError,
    DataCheckMessageCode,
    DCAOParameterType,
)
from evalml.utils import infer_feature_types


class DateTimeFormatDataCheck(DataCheck):
    """Check if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order to be supported by time series estimators.

    Args:
        datetime_column (str, int): The name of the datetime column. If the datetime values are in the index, then pass "index".
    """

    def __init__(self, datetime_column="index"):
        self.datetime_column = datetime_column

    def validate(self, X, y):
        """Checks if the target data has equal intervals and is monotonically increasing.

        Will return a DataCheckError if the data is not a datetime type, is not increasing, has redundant or missing row(s),
        contains invalid (NaN or None) values, or has values that don't align with the assumed frequency.

        Args:
            X (pd.DataFrame, np.ndarray): Features.
            y (pd.Series, np.ndarray): Target data.

        Returns:
            dict (DataCheckError): List with DataCheckErrors if unequal intervals are found in the datetime column.

        Examples:
            >>> import pandas as pd

            The column 'dates' has a set of two dates with daily frequency, two dates with hourly frequency,
            and two dates with monthly frequency.

            >>> X = pd.DataFrame(pd.date_range("2015-01-01", periods=2).append(pd.date_range("2015-01-08", periods=2, freq="H").append(pd.date_range("2016-03-02", periods=2, freq="M"))), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0])
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "No frequency could be detected in column 'dates', possibly due to uneven intervals.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_NO_FREQUENCY_INFERRED",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      }
            ... ]

            The column "dates" has a gap in the values, which implies there are many dates missing.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=9).append(pd.date_range("2021-01-31", periods=50)), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> ww_payload = infer_frequency(X["dates"], debug=True, window_length=5, threshold=0.8)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Column 'dates' has datetime values missing between start and end date.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_IS_MISSING_VALUES",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      },
            ...     {
            ...         "message": "A frequency was detected in column 'dates', but there are faulty datetime values that need to be addressed.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {'columns': None, 'rows': None},
            ...         "action_options": [
            ...             {
            ...                 'code': 'REGULARIZE_AND_IMPUTE_DATASET',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'metadata': {
            ...                         'columns': None,
            ...                         'is_target': True,
            ...                         'rows': None
            ...                 },
            ...                 'parameters': {
            ...                         'time_index': {
            ...                             'default_value': 'dates',
            ...                             'parameter_type': 'global',
            ...                             'type': 'str'
            ...                         },
            ...                         'frequency_payload': {
            ...                             'default_value': ww_payload,
            ...                             'parameter_type': 'global',
            ...                             'type': 'tuple'
            ...                         }
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]

            The column "dates" has a repeat of the date 2021-01-09 appended to the end, which is considered redundant and will raise an error.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", periods=9).append(pd.date_range("2021-01-09", periods=1)), columns=["dates"])
            >>> y = pd.Series([0, 1, 0, 1, 1, 0, 0, 0, 1, 0])
            >>> ww_payload = infer_frequency(X["dates"], debug=True, window_length=5, threshold=0.8)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="dates")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Column 'dates' has more than one row with the same datetime value.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_REDUNDANT_ROW",
            ...         "details": {"columns": None, "rows": None},
            ...         "action_options": []
            ...      },
            ...     {
            ...         "message": "A frequency was detected in column 'dates', but there are faulty datetime values that need to be addressed.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {'columns': None, 'rows': None},
            ...         "action_options": [
            ...             {
            ...                 'code': 'REGULARIZE_AND_IMPUTE_DATASET',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'metadata': {
            ...                         'columns': None,
            ...                         'is_target': True,
            ...                         'rows': None
            ...                 },
            ...                 'parameters': {
            ...                         'time_index': {
            ...                             'default_value': 'dates',
            ...                             'parameter_type': 'global',
            ...                             'type': 'str'
            ...                         },
            ...                         'frequency_payload': {
            ...                             'default_value': ww_payload,
            ...                             'parameter_type': 'global',
            ...                             'type': 'tuple'
            ...                         }
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]

            The column "Weeks" has a date that does not follow the weekly pattern, which is considered misaligned.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", freq="W", periods=12).append(pd.date_range("2021-03-22", periods=1)), columns=["Weeks"])
            >>> ww_payload = infer_frequency(X["Weeks"], debug=True, window_length=5, threshold=0.8)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Column 'Weeks' has datetime values that do not align with the inferred frequency.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None},
            ...         "code": "DATETIME_HAS_MISALIGNED_VALUES",
            ...         "action_options": []
            ...      },
            ...     {
            ...         "message": "A frequency was detected in column 'Weeks', but there are faulty datetime values that need to be addressed.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {'columns': None, 'rows': None},
            ...         "action_options": [
            ...             {
            ...                 'code': 'REGULARIZE_AND_IMPUTE_DATASET',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'metadata': {
            ...                         'columns': None,
            ...                         'is_target': True,
            ...                         'rows': None
            ...                 },
            ...                 'parameters': {
            ...                         'time_index': {
            ...                             'default_value': 'Weeks',
            ...                             'parameter_type': 'global',
            ...                             'type': 'str'
            ...                         },
            ...                         'frequency_payload': {
            ...                             'default_value': ww_payload,
            ...                             'parameter_type': 'global',
            ...                             'type': 'tuple'
            ...                         }
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]

            The column "Weeks" has a date that does not follow the weekly pattern, which is considered misaligned.

            >>> X = pd.DataFrame(pd.date_range("2021-01-01", freq="W", periods=12).append(pd.date_range("2021-03-22", periods=1)), columns=["Weeks"])
            >>> ww_payload = infer_frequency(X["Weeks"], debug=True, window_length=5, threshold=0.8)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="Weeks")
            >>> assert datetime_format_dc.validate(X, y) == [
            ...     {
            ...         "message": "Column 'Weeks' has datetime values that do not align with the inferred frequency.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None},
            ...         "code": "DATETIME_HAS_MISALIGNED_VALUES",
            ...         "action_options": []
            ...      },
            ...     {
            ...         "message": "A frequency was detected in column 'Weeks', but there are faulty datetime values that need to be addressed.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {'columns': None, 'rows': None},
            ...         "action_options": [
            ...             {
            ...                 'code': 'REGULARIZE_AND_IMPUTE_DATASET',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'metadata': {
            ...                         'columns': None,
            ...                         'is_target': True,
            ...                         'rows': None
            ...                 },
            ...                 'parameters': {
            ...                         'time_index': {
            ...                             'default_value': 'Weeks',
            ...                             'parameter_type': 'global',
            ...                             'type': 'str'
            ...                         },
            ...                         'frequency_payload': {
            ...                             'default_value': ww_payload,
            ...                             'parameter_type': 'global',
            ...                             'type': 'tuple'
            ...                         }
            ...                 }
            ...             }
            ...         ]
            ...     }
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

            The first value in the column "index" is replaced with NaT, which will raise an error in this data check.

            >>> dates = [["2-1-21", "3-1-21"],
            ...         ["2-2-21", "3-2-21"],
            ...         ["2-3-21", "3-3-21"],
            ...         ["2-4-21", "3-4-21"],
            ...         ["2-5-21", "3-5-21"],
            ...         ["2-6-21", "3-6-21"],
            ...         ["2-7-21", "3-7-21"],
            ...         ["2-8-21", "3-8-21"],
            ...         ["2-9-21", "3-9-21"],
            ...         ["2-10-21", "3-10-21"],
            ...         ["2-11-21", "3-11-21"],
            ...         ["2-12-21", "3-12-21"]]
            >>> dates[0][0] = None
            >>> df = pd.DataFrame(dates, columns=["days", "days2"])
            >>> ww_payload = infer_frequency(pd.to_datetime(df["days"]), debug=True, window_length=5, threshold=0.8)
            >>> datetime_format_dc = DateTimeFormatDataCheck(datetime_column="days")
            >>> assert datetime_format_dc.validate(df, y) == [
            ...     {
            ...         "message": "Input datetime column 'days' contains NaN values. Please impute NaN values or drop these rows.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "details": {"columns": None, "rows": None},
            ...         "code": "DATETIME_HAS_NAN",
            ...         "action_options": []
            ...      },
            ...     {
            ...         "message": "A frequency was detected in column 'days', but there are faulty datetime values that need to be addressed.",
            ...         "data_check_name": "DateTimeFormatDataCheck",
            ...         "level": "error",
            ...         "code": "DATETIME_HAS_UNEVEN_INTERVALS",
            ...         "details": {'columns': None, 'rows': None},
            ...         "action_options": [
            ...             {
            ...                 'code': 'REGULARIZE_AND_IMPUTE_DATASET',
            ...                 'data_check_name': 'DateTimeFormatDataCheck',
            ...                 'metadata': {
            ...                         'columns': None,
            ...                         'is_target': True,
            ...                         'rows': None
            ...                 },
            ...                 'parameters': {
            ...                         'time_index': {
            ...                             'default_value': 'days',
            ...                             'parameter_type': 'global',
            ...                             'type': 'str'
            ...                         },
            ...                         'frequency_payload': {
            ...                             'default_value': ww_payload,
            ...                             'parameter_type': 'global',
            ...                             'type': 'tuple'
            ...                         }
            ...                 }
            ...             }
            ...         ]
            ...     }
            ... ]
            ...
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
                ).to_dict(),
            )
            return messages

        # Check if the data is monotonically increasing
        no_nan_datetime_values = datetime_values.dropna()
        if not pd.DatetimeIndex(no_nan_datetime_values).is_monotonic_increasing:
            messages.append(
                DataCheckError(
                    message="Datetime values must be sorted in ascending order.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                ).to_dict(),
            )

        col_name = (
            self.datetime_column if self.datetime_column != "index" else "either index"
        )

        ww_payload = infer_frequency(
            pd.Series(datetime_values),
            debug=True,
            window_length=4,
            threshold=0.4,
        )
        inferred_freq = ww_payload[0]
        debug_object = ww_payload[1]
        if inferred_freq is not None:
            return messages

        # Check for NaN values
        if len(debug_object["nan_values"]) > 0:
            messages.append(
                DataCheckError(
                    message=f"Input datetime column '{col_name}' contains NaN values. Please impute NaN values or drop these rows.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_NAN,
                ).to_dict(),
            )

        # Check for only one row per datetime
        if len(debug_object["duplicate_values"]) > 0:
            messages.append(
                DataCheckError(
                    message=f"Column '{col_name}' has more than one row with the same datetime value.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_REDUNDANT_ROW,
                ).to_dict(),
            )

        # Check for no date missing in ordered dates
        if len(debug_object["missing_values"]) > 0:
            messages.append(
                DataCheckError(
                    message=f"Column '{col_name}' has datetime values missing between start and end date.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_IS_MISSING_VALUES,
                ).to_dict(),
            )

        # Check for dates that don't line up with the frequency
        if len(debug_object["extra_values"]) > 0:
            messages.append(
                DataCheckError(
                    message=f"Column '{col_name}' has datetime values that do not align with the inferred frequency.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_MISALIGNED_VALUES,
                ).to_dict(),
            )

        # Give a generic uneven interval error no frequency can be estimated by woodwork
        if debug_object["estimated_freq"] is None:
            messages.append(
                DataCheckError(
                    message=f"No frequency could be detected in column '{col_name}', possibly due to uneven intervals.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_NO_FREQUENCY_INFERRED,
                ).to_dict(),
            )
        else:
            messages.append(
                DataCheckError(
                    message=f"A frequency was detected in column '{col_name}', but there are faulty datetime values that need to be addressed.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                    action_options=[
                        DataCheckActionOption(
                            DataCheckActionCode.REGULARIZE_AND_IMPUTE_DATASET,
                            data_check_name=self.name,
                            parameters={
                                "time_index": {
                                    "parameter_type": DCAOParameterType.GLOBAL,
                                    "type": "str",
                                    "default_value": col_name,
                                },
                                "frequency_payload": {
                                    "parameter_type": DCAOParameterType.GLOBAL,
                                    "type": "tuple",
                                    "default_value": ww_payload,
                                },
                            },
                            metadata={"is_target": True},
                        ),
                    ],
                ).to_dict(),
            )

        return messages
