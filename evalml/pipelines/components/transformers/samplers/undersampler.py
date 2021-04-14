import pandas as pd

from evalml.pipelines.components.transformers.samplers.base_sampler import (
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
        sampler = BalancedClassificationSampler(**parameters, random_seed=random_seed)

        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        X, y, X_pd, y_pd = self._prepare_data(X, y)
        index_df = pd.Series(y_pd.index)
        indices = self._component_obj.fit_resample(X_pd, y_pd)
        train_indices = index_df[index_df.isin(indices)].index.values.tolist()
        return X.iloc[train_indices], y.iloc[train_indices]
