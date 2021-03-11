from evalml.preprocessing.data_splitters.base_splitters import (
    BaseSamplingSplitter
)
from evalml.utils import import_or_raise


class KMeansSMOTETVSplit(BaseSamplingSplitter):
    """Splits the data into training and validation sets and balances the training data using K-Means SMOTE.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', k_neighbors=2, test_size=None, random_seed=0, **kwargs):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        self.sampler = im.KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_seed, **kwargs)
        super().__init__(sampler=self.sampler, test_size=test_size, split_type="TV", random_seed=random_seed)


class KMeansSMOTECVSplit(BaseSamplingSplitter):
    """Splits the data into KFold cross validation sets and balances the training data using K-Means SMOTE.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', k_neighbors=2, n_splits=3, shuffle=True, random_seed=0, **kwargs):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        self.sampler = im.KMeansSMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_seed, **kwargs)
        super().__init__(sampler=self.sampler, n_splits=n_splits, shuffle=shuffle, split_type="CV", random_seed=random_seed)
