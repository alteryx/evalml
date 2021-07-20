import pandas as pd
from sklearn.preprocessing import StandardScaler as SkScaler
from woodwork.logical_types import Boolean, Categorical, Integer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types,
)


class StandardScaler(Transformer):
    """A transformer that standardizes input features by removing the mean and scaling to unit variance.

    Arguments:
        random_seed (int): Seed for the random number generator. Defaults to 0.

    """

    name = "Standard Scaler"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        scaler = SkScaler(**parameters)
        super().__init__(
            parameters=parameters, component_obj=scaler, random_seed=random_seed
        )

    def transform(self, X, y=None):
        X = infer_feature_types(X)
        original_ltypes = X.ww.schema.logical_types
        X = X.ww.select_dtypes(exclude=["datetime"])
        X_t = self._component_obj.transform(X)
        X_t_df = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return _retain_custom_types_and_initalize_woodwork(
            original_ltypes, X_t_df, ltypes_to_ignore=[Integer, Categorical, Boolean]
        )

    def fit_transform(self, X, y=None):
        X = infer_feature_types(X)
        X = X.select_dtypes(exclude=["datetime"])
        return self.fit(X, y).transform(X, y)
