from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils import import_or_raise
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper,
    is_all_numeric
)


class RandomUnderSamplerTVSplit(BaseCrossValidator):
    """Split the data into training and validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', test_size=None, replacement=False, random_state=0):
        super().__init__()
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.rus = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=False, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

    def _to_woodwork(self, X, y, to_pandas=True):
        """Convert the data to woodwork datatype"""
        X_ww = _convert_to_woodwork_structure(X)
        if not is_all_numeric(X_ww):
            raise ValueError('Values not all numeric or there are null values provided in the dataset')
        y_ww = _convert_to_woodwork_structure(y)
        if to_pandas:
            X_ww = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
            y_ww = _convert_woodwork_types_wrapper(y_ww.to_series())
        return X_ww, y_ww

    @staticmethod
    def get_n_splits():
        """Returns the number of splits of this object"""
        return 1

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        X, y = self._to_woodwork(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train_resample, y_train_resample = self.rus.fit_resample(X_train, y_train)
        X_train_resample, y_train_resample = self._to_woodwork(X_train_resample, y_train_resample, to_pandas=False)
        X_test, y_test = self._to_woodwork(X_test, y_test, to_pandas=False)
        return iter([((X_train_resample, y_train_resample), (X_test, y_test))])

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation
        """
        X_pd, y_pd = self._to_woodwork(X, y)
        X_transformed, y_transformed = self.rus.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))


class RandomUnderSamplerCVSplit(StratifiedKFold):
    """Split the training data into KFold cross validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', replacement=False, n_splits=3, shuffle=True, random_state=0):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.rus = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=False, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def _to_woodwork(self, X, y, to_pandas=True):
        """Convert the data to woodwork datatype"""
        X_ww = _convert_to_woodwork_structure(X)
        if not is_all_numeric(X_ww):
            raise ValueError('Values not all numeric or there are null values provided in the dataset')
        y_ww = _convert_to_woodwork_structure(y)
        if to_pandas:
            X_ww = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
            y_ww = _convert_woodwork_types_wrapper(y_ww.to_series())
        return X_ww, y_ww

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        X, y = self._to_woodwork(X, y)
        for i, (train_indices, test_indices) in enumerate(super().split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
            X_train_resample, y_train_resample = self.rus.fit_resample(X_train, y_train)
            X_train_resample, y_train_resample = self._to_woodwork(X_train_resample, y_train_resample, to_pandas=False)
            X_test, y_test = self._to_woodwork(X_test, y_test, to_pandas=False)
            yield iter(((X_train_resample, y_train_resample), (X_test, y_test)))

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation
        """
        X_pd, y_pd = self._to_woodwork(X, y)
        X_transformed, y_transformed = self.rus.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))
