import numpy as np
from sklearn.model_selection import StratifiedKFold

from evalml.preprocessing.data_splitters.base_splitters import (
    BaseUnderSamplingSplitter
)
from evalml.preprocessing.data_splitters.sampler_base import SamplerBase
from evalml.preprocessing.data_splitters.training_validation_split import (
    TrainingValidationSplit
)
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class BalancedClassificationSampler(SamplerBase):
    """Class for balanced classification downsampler."""

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, random_seed=0):
        """
        Arguments:
            balanced_ratio (float): The largest majority:minority ratio that is accepted as 'balanced'. For instance, a 4:1 ratio would be
                represented as 4, while a 6:5 ratio is 1.2. Must be greater than or equal to 1 (or 1:1). Defaults to 4.

            min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
                To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
                Must be greater than 0. Defaults to 100.

            min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
                If class percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
                Must be between 0 and 0.5, inclusive. Defaults to 0.1.

            random_seed (int): The seed to use for random sampling. Defaults to 0.
        """
        super().__init__(random_seed=random_seed)
        if balanced_ratio < 1:
            raise ValueError(f"balanced_ratio must be at least 1, but received {balanced_ratio}")
        if min_samples <= 0:
            raise ValueError(f"min_sample must be greater than 0, but received {min_samples}")
        if min_percentage <= 0 or min_percentage > 0.5:
            raise ValueError(f"min_percentage must be between 0 and 0.5, inclusive, but received {min_percentage}")
        self.balanced_ratio = balanced_ratio
        self.min_samples = min_samples
        self.min_percentage = min_percentage
        self.random_state = np.random.RandomState(self.random_seed)

    def _find_ideal_samples(self, y):
        """Returns dictionary of examples to drop for each class if we need to resample.

        Arguments:
            y (pd.Series): Target data passed in

        Returns:
            (dict): dictionary with undersample target class as key, and number of samples to remove as the value.
                If we don't need to resample, returns empty dictionary.
        """
        counts = y.value_counts()
        normalized_counts = y.value_counts(normalize=True)
        class_ratios = normalized_counts.copy().values
        class_ratios /= class_ratios[-1]
        # if no class ratios are greater than what we consider balanced, then the target is balanced
        if all(class_ratios <= self.balanced_ratio):
            return {}
        # if any classes have less than min_samples counts and are less than min_percentage of the total data,
        # then it's severely imbalanced
        if any(counts < self.min_samples) and any(normalized_counts < self.min_percentage):
            return {}
        # otherwise, we are imbalanced enough to perform on this
        undersample_classes = counts[class_ratios > self.balanced_ratio].index.values
        # find goal size, round it down if it's a float
        goal_value = max(int((self.balanced_ratio * counts.values[-1]) // 1), self.min_samples)
        # we don't want to drop less than 0 rows
        drop_values = {k: max(0, counts[k] - goal_value) for k in undersample_classes}
        return {k: v for k, v in drop_values.items() if v != 0}

    def fit_resample(self, X, y):
        """Resampling technique for this sampler.

        Arguments:
            X (pd.DataFrame): Training data to fit and resample
            y (pd.Series): Training data targets to fit and resample

        Returns:
            list: Indices to keep for training data
        """
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        result = self._find_ideal_samples(y)
        indices_to_drop = []
        if len(result):
            # iterate through the classes we need to undersample and remove the number of samples we need to remove
            for key, value in result.items():
                indices = y.index[y == key].values
                indices_to_remove = self.random_state.choice(indices, value, replace=False)
                indices_to_drop.extend(indices_to_remove)
        return list(set(list(y.index.values)).difference(set(indices_to_drop)))


class BalancedClassificationDataTVSplit(BaseUnderSamplingSplitter):
    """Base class for TV and CV split for Balanced Classification Data Sampler"""

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, test_size=None, shuffle=True, random_seed=0):
        self.sampler = BalancedClassificationSampler(balanced_ratio=balanced_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=1, random_seed=random_seed)
        self.splitter = TrainingValidationSplit(test_size=test_size, shuffle=shuffle, random_state=random_seed)

    def split(self, X, y):
        """Splits and returns the labels of the training and testing data using the data sampler provided.
        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split
        Returns:
            tuple(train, test): A tuple containing the resulting train and test indices, post sampling.
        """
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        for train, test in self.splitter.split(X, y):
            X_train, y_train = X.iloc[train], y.iloc[train]
            train_indices = self.sampler.fit_resample(X_train, y_train)
            return iter([(train_indices, test)])


class BalancedClassificationDataCVSplit(BaseUnderSamplingSplitter):
    """Base class for TV and CV split for Balanced Classification Data Sampler"""

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, n_splits=3, shuffle=True, random_seed=0):
        self.sampler = BalancedClassificationSampler(balanced_ratio=balanced_ratio, min_samples=min_samples, min_percentage=min_percentage, random_seed=random_seed)
        super().__init__(sampler=self.sampler, n_splits=n_splits, random_seed=random_seed)
        self.splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed)

    def split(self, X, y):
        """Splits and returns the sampled training data using the data sampler provided.
        Arguments:
                X (ww.DataTable): DataTable of points to split
                y (ww.DataTable): DataColumn of points to split
        Returns:
            tuple(train, test): A tuple containing the resulting train and test indices, post sampling.
        """
        X_ww = infer_feature_types(X)
        y_ww = infer_feature_types(y)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        for train, test in self.splitter.split(X, y):
            X_train, y_train = X.iloc[train], y.iloc[train]
            train_indices = self.sampler.fit_resample(X_train, y_train)
            yield iter([train_indices, test])
