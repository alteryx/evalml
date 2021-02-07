from evalml.preprocessing.data_splitters.base_splitters import (
    BaseCVSplit,
    BaseTVSplit
)
from evalml.utils import import_or_raise


class KMeansSMOTETVSplit(BaseTVSplit):
    """Splits the data into training and validation sets and balances the training data using K-Means SMOTE.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', k_neighbors=2, test_size=None, random_state=0, **kwargs):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        self.kmsmote = im.KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state, **kwargs)
        super().__init__(sampler=self.kmsmote, test_size=test_size, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y=None):
        """Divides the data into training and testing sets.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        return super().fix_data(X, y)

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation.
        """
        return super().transform_data(X, y)


class KMeansSMOTECVSplit(BaseCVSplit):
    """Split the data into KFold cross validation sets and balances the training data using K-Means SMOTE.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', k_neighbors=2, n_splits=3, shuffle=True, random_state=0, **kwargs):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        self.kmsmote = im.KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state, **kwargs)
        super().__init__(sampler=self.kmsmote, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, X, y=None):
        """Divides the data into training and testing sets.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(list): A tuple containing the resulting X_train, X_valid, y_train, y_valid data.
        """
        for data in super().fix_data(X, y):
            yield data

    def transform(self, X, y):
        """Transforms the input data with the balancing strategy.

            Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split

            Returns:
                tuple(ww.DataTable, ww.DataColumn): A tuple containing the resulting X and y post-transformation.
        """
        return super().transform_data(X, y)
