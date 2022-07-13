"""Data check that checks whether the time series training and validation splits have adequate class representation."""
from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit

from evalml.data_checks import DataCheck, DataCheckError, DataCheckMessageCode
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import infer_feature_types


class TimeSeriesSplittingDataCheck(DataCheck):
    """Checks whether the time series target data is compatible with splitting.

    If the target data in the training and validation of every split doesn't have representation from
    all classes (for time series classification problems) this will prevent the estimators from training
    on all potential outcomes which will cause errors during prediction.

    Args:
        problem_type (str or ProblemTypes): Problem type.
        n_splits (int): Number of time series splits.
    """

    def __init__(self, problem_type, n_splits):
        self.problem_type = problem_type
        if handle_problem_types(self.problem_type) not in [
            ProblemTypes.TIME_SERIES_BINARY,
            ProblemTypes.TIME_SERIES_MULTICLASS,
        ]:
            raise ValueError(
                "Valid splitting of labels in time series is only defined for time series binary and time series multiclass problem types.",
            )
        self.n_splits = n_splits
        self._splitter = SkTimeSeriesSplit(n_splits=self.n_splits)

    def validate(self, X, y):
        """Check if the training and validation targets are compatible with time series data splitting.

        Args:
            X (pd.DataFrame, np.ndarray): Ignored. Features.
            y (pd.Series, np.ndarray): Target data.

        Returns:
            dict: dict with a DataCheckError if splitting would result in inadequate class representation.

        Example:
            >>> import pandas as pd

            Passing n_splits as 3 means that the data will be segmented into 4 parts to be iterated over for training
            and validation splits. The first split results in training indices of [0:25] and validation indices of [25:50].
            The training indices of the first split result in only one unique value (0).
            The third split results in training indices of [0:75] and validation indices of [75:100]. The validation indices
            of the third split result in only one unique value (1).

            >>> X = None
            >>> y = pd.Series([0 if i < 45 else i % 2 if i < 55 else 1 for i in range(100)])
            >>> ts_splitting_check = TimeSeriesSplittingDataCheck("time series binary", 3)
            >>> assert ts_splitting_check.validate(X, y) == [
            ...     {
            ...         "message": "Time Series Binary and Time Series Multiclass problem "
            ...                    "types require every training and validation split to "
            ...                    "have at least one instance of all the target classes. "
            ...                    "The following splits are invalid: [1, 3]",
            ...         "data_check_name": "TimeSeriesSplittingDataCheck",
            ...         "level": "error",
            ...         "details": {
            ...             "columns": None, "rows": None,
            ...             "invalid_splits": {
            ...                 1: {"Training": [0, 25]},
            ...                 3: {"Validation": [75, 100]}
            ...             }
            ...         },
            ...         "code": "TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT",
            ...         "action_options": []
            ...     }
            ... ]
        """
        messages = []

        y = infer_feature_types(y)

        invalid_splits = {}
        y_unique = y.nunique()
        if y is not None:
            for split_num, (train, val) in enumerate(self._splitter.split(X=y)):
                invalid_dict = {}
                train_targets = y[train]
                val_targets = y[val]
                if train_targets.nunique() < y_unique:
                    invalid_dict["Training"] = [0, len(train)]
                if val_targets.nunique() < y_unique:
                    invalid_dict["Validation"] = [len(train), len(train) + len(val)]
                if invalid_dict:
                    invalid_splits[(split_num + 1)] = invalid_dict

        if invalid_splits:
            messages.append(
                DataCheckError(
                    message=f"Time Series Binary and Time Series Multiclass problem types require every training "
                    f"and validation split to have at least one instance of all the target classes. "
                    f"The following splits are invalid: {list(invalid_splits)}",
                    data_check_name=self.name,
                    message_code=DataCheckMessageCode.TIMESERIES_TARGET_NOT_COMPATIBLE_WITH_SPLIT,
                    details={
                        "invalid_splits": invalid_splits,
                    },
                ).to_dict(),
            )
        return messages
