from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils import _convert_numeric_dataset_pandas


class BaseTVSplit(BaseCrossValidator):
    """Base class for training validation data splitter."""

    def __init__(self, sampler=None, test_size=None, random_seed=0):
        """Create a training-validation data splitter instance

        Arguments:
            sampler (sampler instance): The sampler instance to use for resampling the training data. Must have a `fit_resample` method. Defaults to None, which is equivalent to regular TV split.

            test_size (float): What percentage of data points should be included in the validation
                set. Defalts to the complement of `train_size` if `train_size` is set, and 0.25 otherwise.

            random_seed (int): Random seed for the splitter. Defaults to 0
        """
        self.sampler = sampler
        self.test_size = test_size
        self.random_seed = random_seed

    @staticmethod
    def get_n_splits():
        """Returns the number of splits of this object."""
        return 1

    def split(self, X, y):
        """Splits and returns the sampled training data using the data sampler provided.

        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

        Returns:
            tuple((pd.DataFrame, pd.Series), (pd.DataFrame, pd.Series)): A tuple containing the resulting ((X_train, y_train), (X_test, y_test)) post-transformation.
        """
        X, y = _convert_numeric_dataset_pandas(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_seed)
        if self.sampler is not None:
            X_train, y_train = self.sampler.fit_resample(X_train, y_train)
        return iter([((X_train, y_train), (X_test, y_test))])

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(pd.DataFrame, pd.Series): A tuple containing the resulting X and y post-transformation.
        """
        X_pd, y_pd = _convert_numeric_dataset_pandas(X, y)
        X_transformed, y_transformed = self.sampler.fit_resample(X_pd, y_pd)
        return (X_transformed, y_transformed)


class BaseCVSplit(StratifiedKFold):
    """Base class for K-fold cross-validation data splitter."""

    def __init__(self, sampler=None, n_splits=3, shuffle=True, random_seed=0):
        """Create a cross-validation data splitter instance

        Arguments:
            sampler (sampler instance): The sampler instance to use for resampling the training data. Must have a `fit_resample` method. Defaults to None, which is equal to regular K-fold CV.

            n_splits (int): How many CV folds to use. Defaults to 3.

            shuffle (bool): Whether or not to shuffle the data. Defaults to True.

            random_seed (int): Random seed for the splitter. Defaults to 0
        """
        self.sampler = sampler
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)

    def split(self, X, y):
        """Splits using K-fold cross-validation and returns the sampled training data using the data sampler provided.

        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

        Returns:
            tuple((pd.DataFrame, pd.Series), (pd.DataFrame, pd.Series)): An iterator containing the resulting ((X_train, y_train), (X_test, y_test)) post-transformation.
        """
        X, y = _convert_numeric_dataset_pandas(X, y)
        for i, (train_indices, test_indices) in enumerate(super().split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
            if self.sampler is not None:
                X_train, y_train = self.sampler.fit_resample(X_train, y_train)
            yield iter(((X_train, y_train), (X_test, y_test)))

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(pd.DataFrame, pd.Series): A tuple containing the resulting X and y post-transformation.
        """
        X_pd, y_pd = _convert_numeric_dataset_pandas(X, y)
        X_transformed, y_transformed = self.sampler.fit_resample(X_pd, y_pd)
        return (X_transformed, y_transformed)
