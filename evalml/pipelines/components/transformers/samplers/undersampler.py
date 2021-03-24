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

    def __init__(self, balanced_ratio=4, min_samples=100, min_percentage=0.1, random_seed=0):
        parameters = {"balanced_ratio": balanced_ratio,
                      "min_samples": min_samples,
                      "min_percentage": min_percentage,
                      "random_seed": random_seed}
        sampler = BalancedClassificationSampler(**parameters)

        super().__init__(parameters=parameters,
                         component_obj=sampler,
                         random_seed=random_seed)

    def fit(self, X, y):
        """Resample the data using the undersampler. Since our sampler doesn't need to be fit, we do nothing here.

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features
        """
        return self

    def fit_transform(self, X, y):
        """Fit and transform the data using the undersampler. Used during training of the pipeline

        Arguments:
            X (ww.DataFrame): Training features
            y (ww.DataColumn): Target features

        Returns:
            tuple (X, y): Resampled X and y data
        """
        X = infer_feature_types(X)
        y = infer_feature_types(y)
        X_pd = _convert_woodwork_types_wrapper(X.to_dataframe())
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        index_df = pd.Series(y_pd.index)
        indices = self._component_obj.fit_resample(X_pd, y_pd)
        train_indices = index_df[index_df.isin(indices)].dropna().index.values.tolist()
        return (X.iloc[train_indices], y.iloc[train_indices])

    def transform(self, X, y=None):
        """No transformation needs to be done here.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        return X, y
