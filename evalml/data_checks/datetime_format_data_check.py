import pandas as pd

from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.utils import infer_feature_types


class DateTimeFormatDataCheck(DataCheck):
    """Checks if the datetime column has equally spaced intervals and is monotonically increasing or decreasing in order
    to be supported by time series estimators.

    Arguments:
        datetime_column (str): The name of the datetime column. If the datetime values are in the index, then pass "index".

    """

    def __init__(self, datetime_column="index"):
        self.datetime_column = datetime_column

    def validate(self, X, y):
        """Checks if the target data has equal intervals and is sorted.

        Arguments:
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
            >>> assert datetime_format_check.validate(X, y) == {"errors": [{"message": "No frequency could be detected in dates, possibly due to uneven intervals.",\
                                                                    "data_check_name": "EqualIntervalDataCheck",\
                                                                    "level": "error",\
                                                                    "code": "DATETIME_HAS_UNEVEN_INTERVALS",\
                                                                    "details": {}}],\
                                                        "warnings": [],\
                                                        "actions": []}
        """
        results = {"warnings": [], "errors": [], "actions": []}

        X = infer_feature_types(X)
        y = infer_feature_types(y)

        if self.datetime_column != "index":
            datetime_values = X[self.datetime_column]
        else:
            datetime_values = X.index
            if not isinstance(datetime_values, pd.DatetimeIndex):
                datetime_values = y.index
            if not isinstance(datetime_values, pd.DatetimeIndex):
                raise TypeError(
                    "Either X or y has to have datetime information in its index."
                )

        try:
            inferred_freq = pd.infer_freq(datetime_values)
        except TypeError:
            raise TypeError(
                "That column does not contain datetime information or is not in a supported datetime format."
            )

        if not inferred_freq:
            message = (
                self.datetime_column
                if self.datetime_column != "index"
                else "either index"
            )
            results["errors"].append(
                DataCheckError(
                    message=f"No frequency could be detected in {message}, possibly due to uneven intervals.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_HAS_UNEVEN_INTERVALS,
                ).to_dict()
            )

        if not (
            pd.DatetimeIndex(datetime_values).is_monotonic_increasing
            or pd.DatetimeIndex(datetime_values).is_monotonic_decreasing
        ):
            results["errors"].append(
                DataCheckError(
                    message="Datetime values must be monotonically increasing or decreasing.",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.DATETIME_IS_NOT_MONOTONIC,
                ).to_dict()
            )

        return results
