from imblearn.combine import SMOTETomek
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection._split import BaseCrossValidator

from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class SMOTETomekTVSplit(BaseCrossValidator):
    """Split the training data into training and validation sets. Uses SMOTE + Tomek's link to balance the training data,
       but keeps the validation data the same"""

    def __init__(self, sampling_strategy='auto', test_size=None, n_jobs=-1, random_state=0):
        super().__init__()
        self.stl = SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

    def _to_woodwork(self, X, y, to_pandas=True):
        """Convert the data to woodwork datatype"""
        X_ww = _convert_to_woodwork_structure(X)
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
                X (pd.DataFrame): Dataframe of points to split
                y (pd.Series): Series of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        X, y = self._to_woodwork(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train_resample, y_train_resample = self.stl.fit_resample(X_train, y_train)
        X_train_resample, y_train_resample = self._to_woodwork(X_train_resample, y_train_resample, to_pandas=False)
        X_test, y_test = self._to_woodwork(X_test, y_test, to_pandas=False)
        return iter([((X_train_resample, y_train_resample), (X_test, y_test))])

    def transform(self, X, y):
        X_pd, y_pd = self._to_woodwork(X, y)
        X_transformed, y_transformed = self.stl.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))


class SMOTETomekCVSplit(StratifiedKFold):
    """Split the training data into KFold cross validation sets. Uses SMOTE + Tomek's link to balance the training data,
       but keeps the validation data the same"""

    def __init__(self, sampling_strategy='auto', n_splits=3, shuffle=True, n_jobs=-1, random_state=0):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stl = SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def _to_woodwork(self, X, y, to_pandas=True):
        """Convert the data to woodwork datatype"""
        X_ww = _convert_to_woodwork_structure(X)
        y_ww = _convert_to_woodwork_structure(y)
        if to_pandas:
            X_ww = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
            y_ww = _convert_woodwork_types_wrapper(y_ww.to_series())
        return X_ww, y_ww

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (pd.DataFrame): Dataframe of points to split
                y (pd.Series): Series of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        X, y = self._to_woodwork(X, y)
        for i, (train_indices, test_indices) in enumerate(super().split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
            X_train_resample, y_train_resample = self.stl.fit_resample(X_train, y_train)
            X_train_resample, y_train_resample = self._to_woodwork(X_train_resample, y_train_resample, to_pandas=False)
            X_test, y_test = self._to_woodwork(X_test, y_test, to_pandas=False)
            yield iter(((X_train_resample, y_train_resample), (X_test, y_test)))

    def transform(self, X, y):
        X_pd, y_pd = self._to_woodwork(X, y)
        X_transformed, y_transformed = self.stl.fit_resample(X_pd, y_pd)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))
