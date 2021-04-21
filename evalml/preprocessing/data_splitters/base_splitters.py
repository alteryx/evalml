import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils import _convert_numeric_dataset_pandas
from evalml.utils.woodwork_utils import infer_feature_types


class BaseSamplingSplitter(BaseCrossValidator):
    """Base class for training validation and cv data splitter."""

    def __init__(self, sampler=None, test_size=None, n_splits=3, shuffle=True, split_type="TV", random_seed=0):
        """Create a TV or CV data splitter instance

        Arguments:
            sampler (sampler instance): The sampler instance to use for resampling the training data. Must have a `fit_resample` method. Defaults to None, which is equivalent to regular TV split.

            test_size (float): What percentage of data points should be included in the validation
                set. Defalts to the complement of `train_size` if `train_size` is set, and 0.25 otherwise.

            n_splits (int): How many CV folds to use. Defaults to 3.

            shuffle (bool): Whether or not to shuffle the data. Defaults to True.

            split_type (str): Whether to use TV or CV split. Defaults to TV.

            random_seed (int): Random seed for the splitter. Defaults to 0
        """
        self.sampler = sampler
        self.test_size = test_size
        self.n_splits = 1 if split_type == "TV" else n_splits
        self.shuffle = shuffle
        self.split_type = split_type
        self.random_seed = random_seed
        self.splitter = None if split_type == "TV" else StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_seed)

    def get_n_splits(self):
        """Returns the number of splits of this object."""
        return self.n_splits

    def split(self, X, y):
        """Splits and returns the sampled training data using the data sampler provided.

        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

        Returns:
            tuple((pd.DataFrame, pd.Series), (pd.DataFrame, pd.Series)): A tuple containing the resulting ((X_train, y_train), (X_test, y_test)) post-transformation.
        """
        X, y = _convert_numeric_dataset_pandas(X, y)
        if self.split_type == "TV":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)
            if self.sampler is not None:
                X_train, y_train = self.sampler.fit_resample(X_train, y_train)
            yield iter([(X_train, y_train), (X_test, y_test)])
        else:
            for i, (train_indices, test_indices) in enumerate(self.splitter.split(X, y)):
                X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
                if self.sampler is not None:
                    X_train, y_train = self.sampler.fit_resample(X_train, y_train)
                yield iter(((X_train, y_train), (X_test, y_test)))

    def transform_sample(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(pd.DataFrame, pd.Series): A tuple containing the resulting X and y post-transformation.
        """
        X_pd, y_pd = _convert_numeric_dataset_pandas(X, y)
        if self.sampler is not None:
            X_pd, y_pd = self.sampler.fit_resample(X_pd, y_pd)
        return (X_pd, y_pd)


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
        X = infer_feature_types(X)
        y = infer_feature_types(y)
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
        y = infer_feature_types(y)
        index_df = pd.Series(y.index)
        train_index_drop = self.sampler.fit_resample(X, y)
        # convert the indices of the y column into index indices of the original pre-split y
        train_indices = index_df[index_df.isin(train_index_drop)].dropna().index.values.tolist()
        return train_indices
