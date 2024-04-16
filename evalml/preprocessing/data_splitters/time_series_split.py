from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils.gen_utils import are_ts_parameters_valid_for_split

class TimeSeriesSplit(BaseCrossValidator):
    """Rolling Origin Cross Validation specifically designed for time series data.

    This splitter adjusts the training and testing indices based on the max_delay and forecast_horizon
    to avoid data leakage and ensure that predictions are realistic given the temporal nature of the data.

    Parameters:
        max_delay (int): Maximum delay used in feature engineering which creates lagged features,
                         potentially introducing NaNs in the process. The splitter recycles the last
                         `max_delay` rows from the previous split as the start of the current split.
        gap (int): The interval between the end of the data used to create the features and the start of
                   the data used for prediction, ensuring no overlap and future data leakage.
        forecast_horizon (int): Specifies the number of time units to forecast which directly affects the
                                size of each test set in the splits.
        time_index (str, optional): Column name of the datetime series used to sort the data. If provided,
                                    ensures the data is split based on the time order.
        n_series (int, optional): Number of series if the dataset includes multiple time series.
        n_splits (int): Number of splits to generate.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> X = pd.DataFrame([i for i in range(10)], columns=["value"])
        >>> y = pd.Series([i for i in range(10)])
        >>> ts_split = TimeSeriesSplit(n_splits=4, forecast_horizon=2)
        >>> for train_index, test_index in ts_split.split(X, y):
        ...     print("TRAIN:", train_index, "TEST:", test_index)
        ...
        TRAIN: [0 1] TEST: [2 3]
        TRAIN: [0 1 2 3] TEST: [4 5]
        TRAIN: [0 1 2 3 4 5] TEST: [6 7]
        TRAIN: [0 1 2 3 4 5 6 7] TEST: [8 9]
    """

    def __init__(self, max_delay=0, gap=0, forecast_horizon=None, time_index=None, n_series=None, n_splits=3):
        self.max_delay = max_delay
        self.gap = gap
        self.forecast_horizon = forecast_horizon or 1  # Default to 1 if None to ensure at least one in forecast
        self.time_index = time_index
        self.n_series = n_series
        self.n_splits = n_splits

        # Calculate test size based on forecast_horizon and number of series
        test_size = self.forecast_horizon * n_series if n_series else self.forecast_horizon

        # Initialize SkTimeSeriesSplit with calculated test size
        self._splitter = SkTimeSeriesSplit(n_splits=self.n_splits, test_size=test_size)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of data splits defined by the initializer."""
        return self.n_splits

    @staticmethod
    def _check_if_empty(data):
        """Check if the dataframe is None or empty."""
        return data is None or data.empty

    @property
    def is_cv(self):
        """Check if this splitter instance performs cross-validation."""
        return self.n_splits > 1

    def split(self, X, y=None, groups=None):
        """Generates indices to split data into training and test set.

        Takes into consideration max_delay and gap to adjust the training and testing indices.

        Args:
            X (pd.DataFrame): The data containing features.
            y (pd.Series): The target variable series.
            groups (ignored): Only kept for compatibility with the sklearn API.

        Yields:
            train (np.ndarray): The indices of the training data.
            test (np.ndarray): The indices of the testing data.
        
        Raises:
            ValueError: If validation checks fail or both X and y are empty.
        """
        if self._check_if_empty(X) and self._check_if_empty(y):
            raise ValueError("Both X and y cannot be None or empty in TimeSeriesSplit.split")
        X_to_use = y if self._check_if_empty(X) else X

        # Validation of time series parameters
        validation_result = are_ts_parameters_valid_for_split(
            self.gap, self.max_delay, self.forecast_horizon, X_to_use.shape[0], self.n_splits)
        if not validation_result.is_valid:
            raise ValueError(validation_result.msg)

        # Generate splits
        for train, test in self._splitter.split(X_to_use, y, groups):
            yield train, test
