import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection._split import BaseCrossValidator
from imblearn.combine import SMOTETomek
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
        if not isinstance(X, pd.DataFrame):
            X = _convert_woodwork_types_wrapper(X.to_dataframe())
            y = _convert_woodwork_types_wrapper(y.to_series())
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        X_train_resample, y_train_resample = self.stl.fit_resample(X_train, y_train)
        X_train_resample = _convert_to_woodwork_structure(X_train_resample)
        X_test = _convert_to_woodwork_structure(X_test)
        y_train_resample = _convert_to_woodwork_structure(y_train_resample)
        y_test = _convert_to_woodwork_structure(y_test)
        return iter([((X_train_resample, y_train_resample), (X_test, y_test))])
    
    def transform(self, X, y):
        X_ww = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_ww = _convert_woodwork_types_wrapper(y.to_series())
        X_transformed, y_transformed = self.stl.fit_resample(X_ww, y_ww)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))



class SMOTETomekCVSplit(StratifiedKFold):
    """Split the training data into KFold cross validation sets. Uses SMOTE + Tomek's link to balance the training data,
       but keeps the validation data the same"""

    def __init__(self, sampling_strategy='auto', n_splits=3, shuffle=True, n_jobs=-1, random_state=0):
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.stl = SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, X, y=None):
        """Divides the data into training and testing sets

            Arguments:
                X (pd.DataFrame): Dataframe of points to split
                y (pd.Series): Series of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data. 
        """
        if not isinstance(X, pd.DataFrame):
            X = _convert_woodwork_types_wrapper(X.to_dataframe())
            y = _convert_woodwork_types_wrapper(y.to_series())
        for i, (train_indices, test_indices) in enumerate(super().split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]
            X_train_resample, y_train_resample = self.stl.fit_resample(X_train, y_train)
            X_train_resample = _convert_to_woodwork_structure(X_train_resample)
            X_test = _convert_to_woodwork_structure(X_test)
            y_train_resample = _convert_to_woodwork_structure(y_train_resample)
            y_test = _convert_to_woodwork_structure(y_test)
            yield iter(((X_train_resample, y_train_resample), (X_test, y_test)))

    def transform(self, X, y):
        X_ww = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_ww = _convert_woodwork_types_wrapper(y.to_series())
        X_transformed, y_transformed = self.stl.fit_resample(X_ww, y_ww)
        return (_convert_to_woodwork_structure(X_transformed), _convert_to_woodwork_structure(y_transformed))