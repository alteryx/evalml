import pandas as pd
from skopt.space import Integer, Real

from evalml.pipelines.components.transformers import Transformer
from evalml.preprocessing.data_splitters.balanced_classification_sampler import (
    BalancedClassificationSampler
)
from evalml.utils.woodwork_utils import (
    _convert_woodwork_types_wrapper,
    infer_feature_types
)


class Undersampler(Transformer):
    """Random undersampler component."""
    name = "Undersampler"
    hyperparameter_ranges = {
        'balanced_ratio': Real(1, 10),
        'min_samples': Integer(50, 1000),
    }

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, random_seed=0, **kwargs):
        """Initializes an undersampling transformer to downsample the majority classes in the dataset.

        Arguments:
            balanced_ratio (float): The largest majority:minority ratio that is accepted as 'balanced'. For instance, a 4:1 ratio would be
                represented as 4, while a 6:5 ratio is 1.2. Must be greater than or equal to 1 (or 1:1). Defaults to 4.
            min_samples (int): The minimum number of samples that we must have for any class, pre or post sampling. If a class must be downsampled, it will not be downsampled past this value.
                To determine severe imbalance, the minority class must occur less often than this and must have a class ratio below min_percentage.
                Must be greater than 0. Defaults to 100.
            min_percentage (float): The minimum percentage of the minimum class to total dataset that we tolerate, as long as it is above min_samples.
                If min_percentage and min_samples are not met, treat this as severely imbalanced, and we will not resample the data.
                Must be between 0 and 0.5, inclusive. Defaults to 0.1.
            random_seed (int): The seed to use for random sampling. Defaults to 0.
        """
        parameters = {"balanced_ratio": balanced_ratio,
                      "min_samples": min_samples,
                      "min_percentage": min_percentage}
        parameters.update(kwargs)
        sampler = BalancedClassificationSampler(**parameters, random_seed=random_seed)

        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Resample the data using the undersampler. Since our sampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            self
        """
        if y is None:
            raise ValueError("y cannot be none")
        return self

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

         Returns:
            ww.DataTable, ww.DataColumn: Undersampled X and y data
        """
        X = infer_feature_types(X)
        if y is None:
            raise ValueError("y cannot be none")
        y = infer_feature_types(y)
        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        index_df = pd.Series(y_pd.index)
        indices = self._component_obj.fit_resample(X_pd, y_pd)
        train_indices = index_df[index_df.isin(indices)].index.values.tolist()
        return X.iloc[train_indices], y.iloc[train_indices]

    def transform(self, X, y=None):
        """No transformation needs to be done here.

        Arguments:
            X (ww.DataFrame): Training features. Ignored.
            y (ww.DataColumn): Target features. Ignored.

        Returns:
            ww.DataTable, ww.DataColumn: X and y data that was passed in.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y
