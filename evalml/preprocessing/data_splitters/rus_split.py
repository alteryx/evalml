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


class RandomUnderSamplerCVSplit(BaseCVSplit):
    """Split the training data into KFold cross validation sets and uses RandomUnderSampler to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', replacement=False, n_splits=3, shuffle=True, random_state=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.under_sampling", error_msg=error_msg)
        self.rus = im.RandomUnderSampler(sampling_strategy=sampling_strategy, replacement=False, random_state=random_state)
        super().__init__(sampler=self.rus, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
