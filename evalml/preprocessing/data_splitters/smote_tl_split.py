from evalml.preprocessing.data_splitters.base_splitters import (
    BaseSamplingSplitter
)
from evalml.utils import import_or_raise


class SMOTETomekTVSplit(BaseSamplingSplitter):
    """Splits the data into training and validation sets and uses SMOTE + Tomek links balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', test_size=None, n_jobs=-1, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.combine", error_msg=error_msg)
        self.sampler = im.SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_seed)
        super().__init__(sampler=self.sampler, test_size=test_size, split_type="TV", random_seed=random_seed)


class SMOTETomekCVSplit(BaseSamplingSplitter):
    """Splits the data into KFold cross validation sets and uses SMOTE + Tomek links to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', n_splits=3, shuffle=True, n_jobs=-1, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.combine", error_msg=error_msg)
        self.sampler = im.SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_seed)
        super().__init__(sampler=self.sampler, n_splits=n_splits, shuffle=shuffle, split_type="CV", random_seed=random_seed)
