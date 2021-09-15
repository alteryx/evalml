"""An undersampling transformer to downsample the majority classes in the dataset."""
import numpy as np
import pandas as pd

from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler,
)
from evalml.utils.woodwork_utils import infer_feature_types


class Undersampler(BaseSampler):
    """Initializes an undersampling transformer to downsample the majority classes in the dataset.

    This component is only run during training and not during predict.

    Args:
        sampling_ratio (float): The smallest minority:majority ratio that is accepted as 'balanced'. For instance, a 1:4 ratio would be
            represented as 0.25, while a 1:1 ratio is 1.0. Must be between 0 and 1, inclusive. Defaults to 0.25.
        sampling_ratio_dict (dict): A dictionary specifying the desired balanced ratio for each target value. For instance, in a binary case where class 1 is the minority, we could specify:
            `sampling_ratio_dict={0: 0.5, 1: 1}`, which means we would undersample class 0 to have twice the number of samples as class 1 (minority:majority ratio = 0.5), and don't sample class 1.
            Overrides sampling_ratio if provided. Defaults to None.
        min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
            To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
            Must be greater than 0. Defaults to 100.
        min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
            If min_percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
            Must be between 0 and 0.5, inclusive. Defaults to 0.1.
        random_seed (int): The seed to use for random sampling. Defaults to 0.

    Raises:
        ValueError: If sampling_ratio is not in the range (0, 1].
        ValueError: If min_sample is not greater than 0.
        ValueError: If min_percentage is not between 0 and 0.5, inclusive.
    """

    name = "Undersampler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(
        self,
        sampling_ratio=0.25,
        sampling_ratio_dict=None,
        min_samples=100,
        min_percentage=0.1,
        random_seed=0,
        **kwargs,
    ):
        if sampling_ratio <= 0 or sampling_ratio > 1:
            raise ValueError(
                f"sampling_ratio must be within (0, 1], but received {sampling_ratio}"
            )
        if min_samples <= 0:
            raise ValueError(
                f"min_sample must be greater than 0, but received {min_samples}"
            )
        if min_percentage <= 0 or min_percentage > 0.5:
            raise ValueError(
                f"min_percentage must be between 0 and 0.5, inclusive, but received {min_percentage}"
            )

        parameters = {
            "sampling_ratio": sampling_ratio,
            "min_samples": min_samples,
            "min_percentage": min_percentage,
            "sampling_ratio_dict": sampling_ratio_dict,
        }
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        self.min_percentage = min_percentage
        self.random_seed = random_seed
        self.random_state = np.random.RandomState(self.random_seed)
        self.sampling_ratio_dict = sampling_ratio_dict or {}

        parameters.update(kwargs)
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def _initialize_sampler(self, X, y):
        """Helper function to initialize the undersampler component object.

        Args:
            X (pd.DataFrame): Ignored.
            y (pd.Series): The target data.
        """
        pass

    def transform(self, X, y=None):
        """Transforms the input data by sampling the data.

        Args:
            X (pd.DataFrame): Training features.
            y (pd.Series): Target.

        Returns:
            pd.DataFrame, pd.Series: Transformed features and target.
        """
        X_ww, y_ww = self._prepare_data(X, y)
        index_df = pd.Series(y_ww.index)
        indices = self.fit_resample(X_ww, y_ww)

        train_indices = index_df[index_df.isin(indices)].index.values.tolist()
        return X_ww.iloc[train_indices], y_ww.iloc[train_indices]

    def _find_ideal_samples(self, y):
        """Returns dictionary of examples to drop for each class if we need to resample.

        Arguments:
            y (pd.Series): Target data passed in.

        Returns:
            dict: Dictionary with undersample target class as key, and number of samples to remove as the value.
                If we don't need to resample, returns empty dictionary.
        """
        counts = y.value_counts()
        normalized_counts = y.value_counts(normalize=True)
        minority_class_count = min(normalized_counts)
        class_ratios = minority_class_count / normalized_counts
        # if no class ratios are larger than what we consider balanced, then the target is balanced
        if all(class_ratios >= self.sampling_ratio):
            return {}
        # if any classes have less than min_samples counts and are less than min_percentage of the total data,
        # then it's severely imbalanced
        if any(counts < self.min_samples) and any(
            normalized_counts < self.min_percentage
        ):
            return {}
        # otherwise, we are imbalanced enough to perform on this
        undersample_classes = counts[class_ratios <= self.sampling_ratio].index.values
        # find goal size, round it down if it's a float
        minority_class = min(counts.values)
        goal_value = max(
            int((minority_class / self.sampling_ratio) // 1), self.min_samples
        )
        # we don't want to drop less than 0 rows
        drop_values = {k: max(0, counts[k] - goal_value) for k in undersample_classes}
        return {k: v for k, v in drop_values.items() if v > 0}

    def _sampling_dict_to_remove_dict(self, y):
        """Turns the sampling dict input into a dict of samples to remove for each target, similar to the return of _find_ideal_samples.

        Arguments:
            y (pd.Series): Training data targets.

        Returns:
            (dict): dictionary with undersample target class as key, and number of samples to remove as the value.
                If we don't need to resample, returns empty dictionary.
        """
        y_dict = y.value_counts().to_dict()
        new_dic = {}
        for k, v in self.sampling_ratio_dict.items():
            new_dic[k] = max(y_dict[k] - v, 0)
        return new_dic

    def fit_resample(self, X, y):
        """Resampling technique for this sampler.

        Arguments:
            X (pd.DataFrame): Training data to fit and resample.
            y (pd.Series): Training data targets to fit and resample.

        Returns:
            list: Indices to keep for training data.
        """
        if self.parameters["sampling_ratio_dict"]:
            self.sampling_ratio_dict = self._convert_dictionary(
                self.parameters["sampling_ratio_dict"], y
            )

        y = infer_feature_types(y)

        if len(self.sampling_ratio_dict):
            result = self._sampling_dict_to_remove_dict(y)
        else:
            result = self._find_ideal_samples(y)
        indices_to_drop = []
        if len(result):
            # iterate through the classes we need to undersample and remove the number of samples we need to remove
            for key, value in result.items():
                indices = y.index[y == key].values
                indices_to_remove = self.random_state.choice(
                    indices, value, replace=False
                )
                indices_to_drop.extend(indices_to_remove)
        # indices of the y series
        original_indices = list(set(y.index.values).difference(set(indices_to_drop)))
        return original_indices
