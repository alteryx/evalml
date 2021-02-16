from evalml.preprocessing.data_splitters.base_splitters import (
    BaseCVSplit,
    BaseTVSplit
)
from evalml.utils import import_or_raise


def _allowed_categorical(categorical_features):
    if categorical_features is None:
        return False
    elif not isinstance(categorical_features, list):
        return False
    elif len(categorical_features) == 0:
        return False
    elif (all(categorical_features) and all([isinstance(c, bool) for c in categorical_features])):
        return False
    return True


class SMOTENCTVSplit(BaseTVSplit):
    """Splits the data into training and validation sets and uses SMOTENC to balance the training data.
       Keeps the validation data the same. Works on numeric and categorical data, but categorical data must be numerical"""

    def __init__(self, categorical_features=None, sampling_strategy='auto', test_size=None, n_jobs=-1, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        if not _allowed_categorical(categorical_features):
            raise ValueError(f"Categorical feature array must be a list with values and must not all be True, recieved {categorical_features}")
        self.categorical_features = categorical_features
        self.snc = im.SMOTENC(categorical_features=self.categorical_features,
                              sampling_strategy=sampling_strategy,
                              n_jobs=n_jobs, random_state=random_seed)
        super().__init__(sampler=self.snc, test_size=test_size, random_seed=random_seed)


class SMOTENCCVSplit(BaseCVSplit):
    """Splits the data into KFold cross validation sets and uses SMOTENC to balance the training data.
       Keeps the validation data the same. Works on numeric and categorical data, but categorical data must be numerical"""

    def __init__(self, categorical_features=None, sampling_strategy='auto', n_splits=3, shuffle=True, n_jobs=-1, random_seed=0):
        error_msg = "imbalanced-learn is not installed. Please install using 'pip install imbalanced-learn'"
        im = import_or_raise("imblearn.over_sampling", error_msg=error_msg)
        if not _allowed_categorical(categorical_features):
            raise ValueError(f"Categorical feature array must be a list with values and must not all be True, recieved {categorical_features}")
        self.categorical_features = categorical_features
        self.snc = im.SMOTENC(categorical_features=self.categorical_features,
                              sampling_strategy=sampling_strategy,
                              n_jobs=n_jobs, random_state=random_seed)
        super().__init__(sampler=self.snc, n_splits=n_splits, shuffle=shuffle, random_seed=random_seed)
