from evalml.preprocessing.data_splitters.base_splitters import (
    BaseSamplingSplitter
)
from evalml.utils import import_or_raise


class RandomUnderSamplerTVSplit(BaseSamplingSplitter):
    """Splits the data into training and validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', test_size=None, replacement=False, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.sampler = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=replacement, random_state=random_seed)
        super().__init__(sampler=self.sampler, test_size=test_size, split_type="TV", random_seed=random_seed)


class RandomUnderSamplerCVSplit(BaseSamplingSplitter):
    """Splits the training data into KFold cross validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', replacement=False, n_splits=3, shuffle=True, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.sampler = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=replacement, random_state=random_seed)
        super().__init__(sampler=self.sampler, n_splits=n_splits, shuffle=shuffle, split_type="CV", random_seed=random_seed)
