import pandas as pd
from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance."""
    name = "Standard Scaler"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0, **kwargs):
        parameters = {}
        parameters.update(kwargs)

        scaler = SkScaler(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=scaler,
                         random_state=random_state)

    def transform(self, X, y=None):

        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        X_cols = X.columns
        X_index = X.index
        X_t = self._component_obj.transform(X)
        X_t_df = pd.DataFrame(X_t, columns=X_cols, index=X_index)
        # import pdb; pdb.set_trace()
        return _convert_to_woodwork_structure(X_t_df)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

    def fit(self, X, y=None):
        X = _convert_to_woodwork_structure(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if y is not None:
            y = _convert_to_woodwork_structure(y)
            y = _convert_woodwork_types_wrapper(y.to_series())
        self._component_obj.fit(X, y)
        return self
