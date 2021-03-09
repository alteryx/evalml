import pandas as pd
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class BaseUnderSamplingSplitter(BaseCrossValidator):
    """Base class for training validation data splitter."""

    def __init__(self, sampler=None, n_splits=3, random_seed=0):
        """Create a TV or CV data splitter instance
        Arguments:
            sampler (sampler instance): The sampler instance to use for resampling the training data. Must have a `fit_resample` method. Defaults to None, which is equivalent to regular TV split.
            n_splits (int): How many CV folds to use. Defaults to 3.
            random_seed (int): Random seed for the splitter. Defaults to 0
        """
        self.sampler = sampler
        self.n_splits = n_splits
        self.random_seed = random_seed

    def get_n_splits(self):
        """Returns the number of splits of this object."""
        return self.n_splits

    def split(self, X, y):
        """Splits and returns the indices of the training and testing using the data sampler provided.
        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split
        Returns:
            tuple(train, test): A tuple containing the resulting train and test indices, post sampling.
        """
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        index_df = pd.Series(y.index)
        for train, test in self.splitter.split(X, y):
            X_train, y_train = X.iloc[train], y.iloc[train]
            train_index_drop = self.sampler.fit_resample(X_train, y_train)
            # convert the indices of the y column into index indices of the original pre-split y
            train_indices = index_df[index_df.isin(train_index_drop)].dropna().index.values.tolist()
            yield iter([train_indices, test])

    def transform_sample(self, X, y):
        """Transforms the input data with the balancing strategy.
            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split
            Returns:
                list: List of indices to keep
        """
        y_ww = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        index_df = pd.Series(y.index)
        train_index_drop = self.sampler.fit_resample(X, y)
        # convert the indices of the y column into index indices of the original pre-split y
        train_indices = index_df[index_df.isin(train_index_drop)].dropna().index.values.tolist()
        return train_indices
