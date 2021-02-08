from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils.gen_utils import _convert_to_woodwork_structure, _convert_numeric_dataset


class BaseTVSplit(BaseCrossValidator):
    """Base class for training validation data splitter."""

    def __init__(self, sampler=None, test_size=None, random_state=0):
        self.sampler = sampler
        self.test_size = test_size
        self.random_state = random_state

    @staticmethod
    def get_n_splits():
        """Returns the number of splits of this object."""
        return 1

    def split(self, X, y):
        """Splits and returns the data using the data sampler provided.

        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

        Returns:
            tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting [(X_train, y_train), (X_test, y_test)] post-transformation.
        """
        X, y = _convert_numeric_datasetX, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train_resample, y_train_resample = self.sampler.fit_resample(X_train, y_train)
        X_train_resample, y_train_resample = _convert_numeric_datasetX_train_resample, y_train_resample, to_pandas=False)
        X_test, y_test = _convert_numeric_datasetX_test, y_test, to_pandas=False)
        return iter([((X_train_resample, y_train_resample), (X_test, y_test))])

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation.
        """
        X_pd, y_pd = _convert_numeric_datasetX, y)
        X_transformed, y_transformed = self.sampler.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))


class BaseCVSplit(StratifiedKFold):
    """Base class for K-fold cross-validation data splitter."""

    def __init__(self, sampler=None, n_splits=3, shuffle=True, random_state=0):
        self.sampler = sampler
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def split(self, X, y):
        """Splits using K-fold cross-validation and returns the data using the data sampler provided.

        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

        Returns:
            tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting [(X_train, y_train), (X_test, y_test)] post-transformation.
        """
        X, y = _convert_numeric_datasetX, y)
        for i, (train_indices, test_indices) in enumerate(super().split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
            X_train_resample, y_train_resample = self.sampler.fit_resample(X_train, y_train)
            X_train_resample, y_train_resample = _convert_numeric_datasetX_train_resample, y_train_resample, to_pandas=False)
            X_test, y_test = _convert_numeric_datasetX_test, y_test, to_pandas=False)
            yield iter(((X_train_resample, y_train_resample), (X_test, y_test)))

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation.
        """
        X_pd, y_pd = _convert_numeric_datasetX, y)
        X_transformed, y_transformed = self.sampler.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))
