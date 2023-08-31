"""Data check that checks if one or more unique series in a multiseres data is a different length than the others."""

from evalml.data_checks import (
    DataCheck,
    DataCheckMessageCode,
    DataCheckWarning,
)


class MismatchedSeriesLengthDataCheck(DataCheck):
    """Check if one or more unique series in a multiseries dataset is of a different length than the others.

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
            X (pd.DataFrame, np.ndarray): The input features to check.
            y (pd.Series): The target. Defaults to None. Ignored.

        Returns:
            dict: A dictionary of features with column name or index and their probability of being ID columns

        Examples:
            >>> import pandas as pd

            For multiseries time series datasets, each seriesID should ideally have the same number of datetime entries as
            each other. If they don't, then a warning will be raised denoting which seriesID have mismatched lengths

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
            >>> xd = [
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
            ...         "message_code": "MISMATCHED_SERIES_LENGTH",
            ...         "action_options": [],
            ...     }
            ... ]
        """
        messages = []
        if self.series_id not in X:
            raise ValueError(
                f"""series_id "{self.series_id}" doesn't match the series_id column of the dataset.""",
            )

        # gets all the number of entries per series_id
        series_id_len = {}
        for id in X[self.series_id].unique():
            series_id_len[id] = len(X[X[self.series_id] == id])

        # get the majority length
        tracker = {}
        for _, v in series_id_len.items():
            if v not in tracker:
                tracker[v] = 0
            else:
                tracker[v] += 1
        majority_len = max(tracker, key=tracker.get)

        # get the series_id's that aren't the majority length
        not_majority = []
        for id in series_id_len:
            if series_id_len[id] != majority_len:
                not_majority.append(id)

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
