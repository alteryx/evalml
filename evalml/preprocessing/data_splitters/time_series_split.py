"""Rolling Origin Cross Validation for time series problems."""
from sklearn.model_selection import TimeSeriesSplit as SkTimeSeriesSplit
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils.gen_utils import are_ts_parameters_valid_for_split


class TimeSeriesSplit(BaseCrossValidator):
    """Rolling Origin Cross Validation for time series problems.

    The max_delay, gap, and forecast_horizon parameters are only used to validate that the requested split size
    is not too small given these parameters.

    Args:
        max_delay (int): Max delay value for feature engineering. Time series pipelines create delayed features
            from existing features. This process will introduce NaNs into the first max_delay number of rows. The
            splitter uses the last max_delay number of rows from the previous split as the first max_delay number
            of rows of the current split to avoid "throwing out" more data than in necessary. Defaults to 0.
        gap (int): Number of time units separating the data used to generate features and the data to forecast on.
            Defaults to 0.
        forecast_horizon (int, None): Number of time units to forecast. Used for parameter validation. If an integer,
            will set the size of the cv splits. Defaults to None.
        time_index (str): Name of the column containing the datetime information used to order the data. Defaults to None.
        n_splits (int): number of data splits to make. Defaults to 3.

    Example:
        >>> import numpy as np
        >>> import pandas as pd
        ...
        >>> X = pd.DataFrame([i for i in range(10)], columns=["First"])
        >>> y = pd.Series([i for i in range(10)])
        ...
        >>> ts_split = TimeSeriesSplit(n_splits=4)
        >>> generator_ = ts_split.split(X, y)
        ...
        >>> first_split = next(generator_)
        >>> assert (first_split[0] == np.array([0, 1])).all()
        >>> assert (first_split[1] == np.array([2, 3])).all()
        ...
        ...
        >>> second_split = next(generator_)
        >>> assert (second_split[0] == np.array([0, 1, 2, 3])).all()
        >>> assert (second_split[1] == np.array([4, 5])).all()
        ...
        ...
        >>> third_split = next(generator_)
        >>> assert (third_split[0] == np.array([0, 1, 2, 3, 4, 5])).all()
        >>> assert (third_split[1] == np.array([6, 7])).all()
        ...
        ...
        >>> fourth_split = next(generator_)
        >>> assert (fourth_split[0] == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all()
        >>> assert (fourth_split[1] == np.array([8, 9])).all()
    """

    def __init__(
        self,
        max_delay=0,
        gap=0,
        forecast_horizon=None,
        time_index=None,
        n_splits=3,
    ):
        self.max_delay = max_delay
        self.gap = gap
        self.forecast_horizon = forecast_horizon if forecast_horizon else 1
        self.time_index = time_index
        self.n_splits = n_splits
        self._splitter = SkTimeSeriesSplit(
            n_splits=n_splits,
            test_size=forecast_horizon,
        )

    def get_n_splits(self, X=None, y=None, groups=None):
        """Get the number of data splits.

        Args:
            X (pd.DataFrame, None): Features to split.
            y (pd.DataFrame, None): Target variable to split. Defaults to None.
            groups: Ignored but kept for compatibility with sklearn API. Defaults to None.

        Returns:
            Number of splits.
        """
        return self._splitter.n_splits

    @staticmethod
    def _check_if_empty(data):
        return data is None or data.empty

    @property
    def is_cv(self):
        """Returns whether or not the data splitter is a cross-validation data splitter.

        Returns:
            bool: If the splitter is a cross-validation data splitter
        """
        return self._splitter.n_splits > 1

    def split(self, X, y=None, groups=None):
        """Get the time series splits.

        X and y are assumed to be sorted in ascending time order.
        This method can handle passing in empty or None X and y data but note that X and y cannot be None or empty
        at the same time.

        Args:
            X (pd.DataFrame, None): Features to split.
            y (pd.DataFrame, None): Target variable to split. Defaults to None.
            groups: Ignored but kept for compatibility with sklearn API. Defaults to None.

        Yields:
            Iterator of (train, test) indices tuples.

        Raises:
            ValueError: If one of the proposed splits would be empty.
        """
        # Sklearn splitters always assume a valid X is passed but we need to support the
        # TimeSeriesPipeline convention of being able to pass in empty X dataframes
        # We'll do this by passing X=y if X is empty
        if self._check_if_empty(X) and self._check_if_empty(y):
            raise ValueError(
                "Both X and y cannot be None or empty in TimeSeriesSplit.split",
            )
        elif self._check_if_empty(X) and not self._check_if_empty(y):
            split_kwargs = dict(X=y, groups=groups)
        else:
            split_kwargs = dict(X=X, y=y, groups=groups)

        result = are_ts_parameters_valid_for_split(
            self.gap,
            self.max_delay,
            self.forecast_horizon,
            X.shape[0],
            self.n_splits,
        )
        if not result.is_valid:
            raise ValueError(result.msg)

        for train, test in self._splitter.split(**split_kwargs):
            yield train, test
