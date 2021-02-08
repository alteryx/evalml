from evalml.preprocessing.data_splitters.base_splitters import (
    BaseCVSplit,
    BaseTVSplit
)
from evalml.utils import import_or_raise


class RandomUnderSamplerTVSplit(BaseTVSplit):
    """Split the data into training and validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', test_size=None, replacement=False, random_state=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.rus = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=replacement, random_state=random_state)
        super().__init__(sampler=self.rus, test_size=test_size, random_state=random_state)
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X, y):
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


class RandomUnderSamplerCVSplit(BaseCVSplit):
    """Split the training data into KFold cross validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', replacement=False, n_splits=3, shuffle=True, random_state=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.rus = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=False, random_state=random_state)
        super().__init__(sampler=self.rus, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.random_state = random_state
        self.n_splits = n_splits

    def split(self, X, y):
        """Divides the data into cross-validation splits.

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
