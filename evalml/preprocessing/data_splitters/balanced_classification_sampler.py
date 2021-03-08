import numpy as np

from evalml.preprocessing.data_splitters.sampler_base import SamplerBase
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
                To determine severe imbalance, the minority class must have a class ratio below this and must occur less often than min_samples.
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
        class_ratios = normalized_counts / min(normalized_counts)
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
        minority_class = min(counts.values)
        goal_value = max(int((self.balanced_ratio * minority_class) // 1), self.min_samples)
        # we don't want to drop less than 0 rows
        drop_values = {k: max(0, counts[k] - goal_value) for k in undersample_classes}
        return {k: v for k, v in drop_values.items() if v > 0}

    def fit_resample(self, X, y):
        """Resampling technique for this sampler.

        Arguments:
            X (pd.DataFrame): Training data to fit and resample
            y (pd.Series): Training data targets to fit and resample

        Returns:
            list: Indices to keep for training data
        """
        y_ww = infer_feature_types(y)
        y = _convert_woodwork_types_wrapper(y_ww.to_series())
        result = self._find_ideal_samples(y)
        indices_to_drop = []
        if len(result):
            # iterate through the classes we need to undersample and remove the number of samples we need to remove
            for key, value in result.items():
                indices = y.index[y == key].values
                indices_to_remove = self.random_state.choice(indices, value, replace=False)
                indices_to_drop.extend(indices_to_remove)
        # indices of the y datacolumn
        original_indices = list(set(y.index.values).difference(set(indices_to_drop)))
        return original_indices
