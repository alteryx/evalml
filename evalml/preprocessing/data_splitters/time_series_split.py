import numpy as np
from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):
    """Rolling Origin Cross Validation for time series problems."""

    def __init__(self, max_delay=0, gap=0, date_index=None, n_splits=3):
        """Create a TimeSeriesSplit.

        This class uses max_delay and gap values to take into account that evalml time series pipelines perform
        some feature and target engineering, e.g delaying input features and shifting the target variable by the
        desired amount. If the data that will be split already has all the features and appropriate target values, and
        then set max_delay and gap to 0.

        Arguments:
            max_delay (int): Max delay value for feature engineering. Time series pipelines create delayed features
                from existing features. This process will introduce NaNs into the first max_delay number of rows. The
                splitter uses the last max_delay number of rows from the previous split as the first max_delay number
                of rows of the current split to avoid "throwing out" more data than in necessary.
            gap (int): Gap used in time series problem. Time series pipelines shift the target variable by gap rows
                since we are interested in
            date_index (str): Name of the column containing the datetime information used to order the data.
            n_splits (int): number of data splits to make.
            """
        self.max_delay = max_delay
        self.gap = gap
        self.date_index = date_index
        self.n_splits = n_splits
        self._splitter = SkTimeSeriesSplit(n_splits=n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of data splits."""
        return self._splitter.n_splits

    @staticmethod
    def _check_if_empty(data):
        return data is None or data.empty

    def split(self, X, y=None, groups=None):
        """Get the time series splits.

        X and y are assumed to be sorted in ascending time order.
        This method can handle passing in empty or None X and y data but note that X and y cannot be None or empty
        at the same time.

        Arguments:
            X (pd.DataFrame, None): Features to split.
            y (pd.DataFrame, None): Target variable to split.
            groups: Ignored but kept for compatibility with sklearn api.

        Returns:
            Iterator of (train, test) indices tuples.
        """
        # Sklearn splitters always assume a valid X is passed but we need to support the
        # TimeSeriesPipeline convention of being able to pass in empty X dataframes
        # We'll do this by passing X=y if X is empty
        if self._check_if_empty(X) and self._check_if_empty(y):
            raise ValueError("Both X and y cannot be None or empty in TimeSeriesSplit.split")
        elif self._check_if_empty(X) and not self._check_if_empty(y):
            split_kwargs = dict(X=y, groups=groups)
            max_index = y.shape[0]
        else:
            split_kwargs = dict(X=X, y=y, groups=groups)
            max_index = X.shape[0]

        split_size = (max_index // self.n_splits)
        if split_size < self.gap + self.max_delay:
            raise ValueError(f"Since the data has {max_index} observations and n_splits={self.n_splits}, "
                             f"the smallest split would have {split_size} observations. "
                             f"Since {self.gap + self.max_delay} (gap + max_delay)  > {split_size}, "
                             "then at least one of the splits would be empty by the time it reaches the pipeline. "
                             "Please use a smaller number of splits or collect more data.")

        for train, test in self._splitter.split(**split_kwargs):
            last_train = train[-1]
            first_test, last_test = test[0], test[-1]

            # Make sure the splits don't extend past our data
            max_test_index = min(last_test + 1 + self.gap, max_index)
            max_train_index = min(last_train + 1 + self.gap, max_index)

            new_train = np.arange(0, max_train_index)
            new_test = np.arange(first_test - self.max_delay, max_test_index)
            yield new_train, new_test
