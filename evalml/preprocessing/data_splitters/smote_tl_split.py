from evalml.preprocessing.data_splitters.base_splitters import (
    BaseCVSplit,
    BaseTVSplit
)
from evalml.utils import import_or_raise


class SMOTETomekTVSplit(BaseTVSplit):
    """Splits the data into training and validation sets and uses SMOTE + Tomek's link bbalance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', test_size=None, n_jobs=-1, random_state=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.combine", error_msg=error_msg)
        self.stl = im.SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_state)
        super().__init__(sampler=self.stl, test_size=test_size, random_state=random_state)


class SMOTETomekCVSplit(BaseCVSplit):
    """Split the data into KFold cross validation sets and uses SMOTE + Tomek's link to balance the training data.
       Keeps the validation data the same. Works only on continuous, numeric data."""

    def __init__(self, sampling_strategy='auto', n_splits=3, shuffle=True, n_jobs=-1, random_state=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.combine", error_msg=error_msg)
        self.stl = im.SMOTETomek(sampling_strategy=sampling_strategy, n_jobs=n_jobs, random_state=random_state)
        super().__init__(sampler=self.stl, n_splits=n_splits, shuffle=shuffle, random_state=random_state)
