import pandas as pd
from sklearn.preprocessing import StandardScaler as SkScaler
from woodwork.logical_types import Categorical, Integer

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance."""
    name = "Standard Scaler"
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        scaler = SkScaler(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=scaler,
                         random_seed=random_seed)

    def transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X = X.select_dtypes(exclude=['datetime'])
        X_t = self._component_obj.transform(X)
        X_t_df = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return _retain_custom_types_and_initalize_woodwork(X_ww, X_t_df, ltypes_to_ignore=[Integer, Categorical])

    def fit_transform(self, X, y=None):
        X_ww = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        X = X.select_dtypes(exclude=['datetime'])
        return self.fit(X, y).transform(X, y)
