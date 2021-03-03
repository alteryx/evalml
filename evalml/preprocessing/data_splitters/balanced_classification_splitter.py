from sklearn.model_selection import StratifiedKFold

from evalml.preprocessing.data_splitters.balanced_classification_sampler import (
    BalancedClassificationSampler
)
from evalml.preprocessing.data_splitters.base_splitters import (
    BaseUnderSamplingSplitter
)
from evalml.preprocessing.data_splitters.training_validation_split import (
    TrainingValidationSplit
)
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class BalancedClassificationDataTVSplit(BaseUnderSamplingSplitter):
    """Data splitter for generating training and validation split using Balanced Classification Data Sampler."""

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, test_size=None, shuffle=True, random_seed=0):
        self.sampler = BalancedClassificationSampler(balanced_ratio=balanced_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=1, random_seed=random_seed)
        self.shuffle = shuffle
        self.splitter = TrainingValidationSplit(test_size=test_size, shuffle=shuffle, random_state=random_seed)


class BalancedClassificationDataCVSplit(BaseUnderSamplingSplitter):
    """Data splitter for generating k-fold cross-validation split using Balanced Classification Data Sampler."""

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, n_splits=3, shuffle=True, random_seed=0):
        self.sampler = BalancedClassificationSampler(balanced_ratio=balanced_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=n_splits, random_seed=random_seed)
        self.shuffle = shuffle
        self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
