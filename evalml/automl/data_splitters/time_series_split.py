import numpy as np
from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator


class TimeSeriesSplit(BaseCrossValidator):
    """Rolling Origin Cross Validation for time series problems."""

    def __init__(self, max_delay=0, gap=0, n_folds=3):
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
            """
        self.max_delay = max_delay
        self.gap = gap
        self.n_folds = n_folds
        self._splitter = SkTimeSeriesSplit(n_splits=n_folds)

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of data splits."""
        return self._splitter.n_splits

    @staticmethod
    def _check_if_empty(data):
        return data is None or data.empty

    def split(self, X, y=None, groups=None):
        """Get the time series splits.

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

        for train, test in self._splitter.split(**split_kwargs):
            last_train = train[-1]
            last_test = test[-1]
            first_test = test[0]
            max_test_index = min(last_test + 1 + self.gap, max_index)
            new_train = np.concatenate([train, np.arange(last_train + 1, last_train + 1 + self.gap)])
            new_test = np.concatenate([
                np.arange(first_test - self.max_delay, first_test), test, np.arange(last_test + 1, max_test_index)
            ])
            yield new_train, new_test
