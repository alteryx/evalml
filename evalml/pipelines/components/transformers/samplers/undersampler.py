import pandas as pd

from evalml.pipelines.components.transformers import (
    BaseSampler
)
from evalml.preprocessing.data_splitters.balanced_classification_sampler import (
    BalancedClassificationSampler
)


class Undersampler(BaseSampler):
    """Random undersampler component. This component is only run during training and not during predict."""
    name = "Undersampler"
    hyperparameter_ranges = {}

    def __init__(self, sampling_ratio=0.25, min_samples=100, min_percentage=0.1, random_seed=0, **kwargs):
        """Initializes an undersampling transformer to downsample the majority classes in the dataset.

        Arguments:
            sampling_ratio (float): The smallest minority:majority ratio that is accepted as 'balanced'. For instance, a 1:4 ratio would be
                represented as 0.25, while a 1:1 ratio is 1.0. Must be between 0 and 1, inclusive. Defaults to 0.25.
            min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
                To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
                Must be greater than 0. Defaults to 100.
            min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
                If min_percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
                Must be between 0 and 0.5, inclusive. Defaults to 0.1.
            random_seed (int): The seed to use for random sampling. Defaults to 0.
        """
        parameters = {"sampling_ratio": sampling_ratio,
                      "min_samples": min_samples,
                      "min_percentage": min_percentage}
        parameters.update(kwargs)
        if sampling_ratio <= 0 or sampling_ratio > 1:
            raise ValueError(f"sampling_ratio must be within (0, 1], but received {sampling_ratio}")
        if min_samples <= 0:
            raise ValueError(f"min_sample must be greater than 0, but received {min_samples}")
        if min_percentage <= 0 or min_percentage > 0.5:
            raise ValueError(f"min_percentage must be between 0 and 0.5, inclusive, but received {min_percentage}")
        self.sampling_ratio = sampling_ratio
        self.min_samples = min_samples
        self.min_percentage = min_percentage
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Since our undersampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (ww.DataFrame): Training features. Ignored for this component.
            y (ww.DataColumn): Target.

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be none")
        return self

    def _find_ideal_samples(self, y):
        """Returns dictionary of examples to drop for each class if we need to resample.

        Arguments:
            y (pd.Series): Target.

        Returns:
            (dict): dictionary with undersample target class as key, and number of samples to remove as the value.
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
        if any(counts < self.min_samples) and any(normalized_counts < self.min_percentage):
            return {}
        # otherwise, we are imbalanced enough to perform on this
        undersample_classes = counts[class_ratios <= self.sampling_ratio].index.values
        # find goal size, round it down if it's a float
        minority_class = min(counts.values)
        goal_value = max(int((minority_class / self.sampling_ratio) // 1), self.min_samples)
        # we don't want to drop less than 0 rows
        drop_values = {k: max(0, counts[k] - goal_value) for k in undersample_classes}
        return {k: v for k, v in drop_values.items() if v > 0}

    def transform(self, X, y):
        """Apply undersampling. Used during training of the pipeline.

        Arguments:
            X (ww.DataFrame): Training features. Ignored for this component.
            y (ww.DataColumn): Target.

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        if y is None:
            raise ValueError("y cannot be none")
        y = infer_feature_types(y)
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        index_df = pd.Series(y.to_series().index)
        random_state = np.random.RandomState(self.random_seed)

        result = self._find_ideal_samples(y_pd)
        indices_to_drop = []
        if len(result):
            # iterate through the classes we need to undersample and remove the number of samples we need to remove
            for key, value in result.items():
                indices = y_pd.index[y_pd == key].values
                indices_to_remove = random_state.choice(indices, value, replace=False)
                indices_to_drop.extend(indices_to_remove)
        indices = list(set(y_pd.index.values).difference(set(indices_to_drop)))
        train_indices = index_df[index_df.isin(indices)].index.values.tolist()
        return X.iloc[train_indices], y.iloc[train_indices]
