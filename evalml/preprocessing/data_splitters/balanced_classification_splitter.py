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


class BalancedClassificationDataTVSplit(BaseUnderSamplingSplitter):
    """Data splitter for generating training and validation split using Balanced Classification Data Sampler."""

    def __init__(self, sampling_ratio=0.25, min_samples=100, min_percentage=0.1, test_size=0.25, shuffle=True, random_seed=0):
        """Create Balanced Classification Data TV splitter

        Arguments:
            sampling_ratio (float): The smallest minority:majority ratio that is accepted as 'balanced'. For instance, a 1:4 ratio would be
                represented as 0.25, while a 1:1 ratio is 1.0. Must be between 0 and 1, inclusive. Defaults to 0.25.

            min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
                To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
                Must be greater than 0. Defaults to 100.

            min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
                If min_percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
                Must be between 0 and 0.5, inclusive. Defaults to 0.1.

            test_size (float): The size of the test split. Defaults to 0.25.

            shuffle (bool): Whether or not to shuffle the data before splitting. Defaults to True.

            random_seed (int): The seed to use for random sampling. Defaults to 0.
        """
        self.sampler = BalancedClassificationSampler(sampling_ratio=sampling_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=1, random_seed=random_seed)
        self.shuffle = shuffle
        self.test_size = test_size
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        self.min_percentage = min_percentage
        self.splitter = TrainingValidationSplit(test_size=test_size, shuffle=shuffle, random_seed=random_seed)


class BalancedClassificationDataCVSplit(BaseUnderSamplingSplitter):
    """Data splitter for generating k-fold cross-validation split using Balanced Classification Data Sampler."""

    def __init__(self, sampling_ratio=0.25, min_samples=100, min_percentage=0.1, n_splits=3, shuffle=True, random_seed=0):
        """Create Balanced Classification Data CV splitter

        Arguments:
            sampling_ratio (float): The smallest minority:majority ratio that is accepted as 'balanced'. For instance, a 1:4 ratio would be
                represented as 0.25, while a 1:1 ratio is 1.0. Must be between 0 and 1, inclusive. Defaults to 0.25.

            min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
                To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
                Must be greater than 0. Defaults to 100.

            min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
                If min_percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
                Must be between 0 and 0.5, inclusive. Defaults to 0.1.

            n_splits (int): The number of splits to use for cross validation. Defaults to 3.

            shuffle (bool): Whether or not to shuffle the data before splitting. Defaults to True.

            random_seed (int): The seed to use for random sampling. Defaults to 0.
        """
        self.sampler = BalancedClassificationSampler(sampling_ratio=sampling_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=n_splits, random_seed=random_seed)
        self.shuffle = shuffle
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        self.min_percentage = min_percentage
        self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)
