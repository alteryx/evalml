import pandas as pd

from evalml.pipelines.components.transformers.samplers.base_sampler import (
    BaseSampler,
)
from evalml.preprocessing.data_splitters.balanced_classification_sampler import (
    BalancedClassificationSampler,
)


class Undersampler(BaseSampler):
    """
    Initializes an undersampling transformer to downsample the majority classes in the dataset.

    This component is only run during training and not during predict.

    Arguments:
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
        parameters = {
            "sampling_ratio": sampling_ratio,
            "min_samples": min_samples,
            "min_percentage": min_percentage,
            "sampling_ratio_dict": sampling_ratio_dict,
        }
        parameters.update(kwargs)
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def _initialize_sampler(self, X, y):
        """Helper function to initialize the undersampler component object.

        Arguments:
            y (pd.Series): The target data
        """
        param_dic = self._dictionary_to_params(
            self.parameters["sampling_ratio_dict"], y
        )
        param_dic.pop("n_jobs", None)
        sampler = BalancedClassificationSampler(
            **param_dic, random_seed=self.random_seed
        )
        self._component_obj = sampler

    def transform(self, X, y=None):
        X_ww, y_ww = self._prepare_data(X, y)
        self._initialize_sampler(X, y_ww)
        index_df = pd.Series(y_ww.index)
        indices = self._component_obj.fit_resample(X_ww, y_ww)

        train_indices = index_df[index_df.isin(indices)].index.values.tolist()
        return X_ww.iloc[train_indices], y_ww.iloc[train_indices]
