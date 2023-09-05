"""Data check that checks if one or more unique series in a multiseres data is a different length than the others."""

from evalml.data_checks import (
    DataCheck,
    DataCheckError,
    DataCheckMessageCode,
    DataCheckWarning,
)


class MismatchedSeriesLengthDataCheck(DataCheck):
    """Check if one or more unique series in a multiseries dataset is of a different length than the others.

    Currently works specifically on stacked data

    Args:
        series_id (str): The name of the series_id column for the dataset.
    """

    def __init__(self, series_id):
        if series_id is None:
            raise ValueError(
                "series_id must be set to the series_id column in the dataset and not None",
            )
        self.series_id = series_id

    def validate(self, X, y=None):
        """Check if one or more unique series in a multiseries dataset is of a different length than the other.

        Currently works specifically on stacked data

        Args:
            X (pd.DataFrame, np.ndarray): The input features to check. Must have a series_id column.
            y (pd.Series): The target. Defaults to None. Ignored.

        Returns:
            dict (DataCheckWarning, DataCheckError): List with DataCheckWarning if there are mismatch series length in the datasets
                  or list with DataCheckError if the given series_id is not in the dataset

        Examples:
            >>> import pandas as pd

            For multiseries time series datasets, each seriesID should ideally have the same number of datetime entries as
            each other. If they don't, then a warning will be raised denoting which seriesID have mismatched lengths.

            >>> X = pd.DataFrame(
            ...     {
            ...         "date": pd.date_range(start="1/1/2018", periods=20).repeat(5),
            ...         "series_id": pd.Series(list(range(5)) * 20, dtype="str"),
            ...         "feature_a": range(100),
            ...         "feature_b": reversed(range(100)),
            ...     },
            ... )
            >>> X = X.drop(labels=0, axis=0)
            >>> mismatched_series_length_check = MismatchedSeriesLengthDataCheck("series_id")
            >>> assert mismatched_series_length_check.validate(X) == [
            ...      {
            ...         "message": "Series ID ['0'] do not match the majority length of the other series, which is 20",
            ...         "data_check_name": "MismatchedSeriesLengthDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": None,
            ...             "rows": None,
            ...             "series_id": ['0'],
            ...             "majority_length": 20
            ...         },
            ...         "code": "MISMATCHED_SERIES_LENGTH",
            ...         "action_options": [],
            ...     }
            ... ]

            If MismatchedSeriesLengthDataCheck is passed in an invalid series_id column name, then an error will be raised.

            >>> X = pd.DataFrame(
            ...     {
            ...         "date": pd.date_range(start="1/1/2018", periods=20).repeat(5),
            ...         "series_id": pd.Series(list(range(5)) * 20, dtype="str"),
            ...         "feature_a": range(100),
            ...         "feature_b": reversed(range(100)),
            ...     },
            ... )
            >>> X = X.drop(labels=0, axis=0)
            >>> mismatched_series_length_check = MismatchedSeriesLengthDataCheck("not_series_id")
            >>> assert mismatched_series_length_check.validate(X) == [
            ...      {
            ...         "message": "series_id 'not_series_id' is not in the dataset.",
            ...         "data_check_name": "MismatchedSeriesLengthDataCheck",
            ...         "level": "error",
            ...         "details": {
            ...             "columns": None,
            ...             "rows": None,
            ...             "series_id": "not_series_id",
            ...         },
            ...         "code": "INVALID_SERIES_ID_COL",
            ...         "action_options": [],
            ...     }
            ... ]

            If there are multiple lengths that have the same number of series (e.g. two series have length 20 and two series have length 19),
            this datacheck will consider the higher length to be the majority length (e.g. from the previous example length 20 would be the majority length)
            >>> X = pd.DataFrame(
            ...     {
            ...         "date": pd.date_range(start="1/1/2018", periods=20).repeat(4),
            ...         "series_id": pd.Series(list(range(4)) * 20, dtype="str"),
            ...         "feature_a": range(80),
            ...         "feature_b": reversed(range(80)),
            ...     },
            ... )
            >>> X = X.drop(labels=[0, 1], axis=0)
            >>> mismatched_series_length_check = MismatchedSeriesLengthDataCheck("series_id")
            >>> assert mismatched_series_length_check.validate(X) == [
            ...      {
            ...         "message": "Series ID ['0', '1'] do not match the majority length of the other series, which is 20",
            ...         "data_check_name": "MismatchedSeriesLengthDataCheck",
            ...         "level": "warning",
            ...         "details": {
            ...             "columns": None,
            ...             "rows": None,
            ...             "series_id": ['0', '1'],
            ...             "majority_length": 20
            ...         },
            ...         "code": "MISMATCHED_SERIES_LENGTH",
            ...         "action_options": [],
            ...     }
            ... ]
        """
        messages = []
        if self.series_id not in X:
            messages.append(
                DataCheckError(
                    message=f"""series_id '{self.series_id}' is not in the dataset.""",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.INVALID_SERIES_ID_COL,
                    details={"series_id": self.series_id},
                ).to_dict(),
            )
            return messages

        # gets all the number of entries per series_id
        series_id_len = {
            id: len(X[X[self.series_id] == id]) for id in X[self.series_id].unique()
        }

        # dictionary where {length: number of series with that length}
        tracker = {}
        for series_length in series_id_len.values():
            if series_length not in tracker:
                tracker[series_length] = 0
            else:
                tracker[series_length] += 1

        if len(tracker) == 1:
            return messages

        majority_len = max(tracker, key=tracker.get)

        # get the series_id's that aren't the majority length
        not_majority = [id for id in series_id_len if series_id_len[id] != majority_len]

        if len(not_majority) > 0 and len(not_majority) < len(series_id_len) - 1:
            warning_msg = f"Series ID {not_majority} do not match the majority length of the other series, which is {majority_len}"
            messages.append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.MISMATCHED_SERIES_LENGTH,
                    details={
                        "series_id": not_majority,
                        "majority_length": majority_len,
                    },
                    action_options=[],
                ).to_dict(),
            )
        elif len(not_majority) == len(series_id_len) - 1:
            warning_msg = "All series ID have different lengths than each other"
            messages.append(
                DataCheckWarning(
                    message=warning_msg,
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.MISMATCHED_SERIES_LENGTH,
                    action_options=[],
                ).to_dict(),
            )
        return messages
