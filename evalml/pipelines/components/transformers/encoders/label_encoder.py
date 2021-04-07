import pandas as pd
import woodwork as ww
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class LabelEncoder(Transformer):
    """Encodes target from string (or other dtype) to integers ranging from 0 to n-1."""
    name = 'Label Encoder'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        self._y_logical_type = None
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        if X is not None:
            X = infer_feature_types(X)
        if y is None:
            return X, None
        y_ww = infer_feature_types(y)
        self._y_logical_type = y_ww.logical_type

        self._encoder = SkLabelEncoder()
        y_pd = _convert_woodwork_types_wrapper(y_ww.to_series())
        self._encoder.fit(y_pd)
        return self

    def transform(self, X, y=None):
        if X is not None:
            X = infer_feature_types(X)
        if y is None:
            return X, None
        y_ww = infer_feature_types(y)

        y_pd = _convert_woodwork_types_wrapper(y_ww.to_series())
        try:
            y_encoded = self._encoder.transform(y_pd)
        except ValueError as e:
            raise ValueError(str(e))
        y_t = pd.Series(y_encoded, index=y_pd.index)
        return X, _retain_custom_types_and_initalize_woodwork(y_ww, y_t)

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


class LabelDecoder(Transformer):
    """Decodes target from integers from 0 to n-1 to string or other original dtype"""
    name = 'Label Decoder'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        """No-op"""
        return self

    def transform(self, X, y=None, dependent_components=None):
        if dependent_components is None or not len(dependent_components) or not isinstance(dependent_components[0], LabelEncoder):
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        label_encoder = dependent_components[0]

        if X is not None:
            X = infer_feature_types(X)
        if y is None:
            return X, None
        y_ww = infer_feature_types(y)
        y_pd = _convert_woodwork_types_wrapper(y_ww.to_series())
        y_decoded = label_encoder._encoder.inverse_transform(y_pd.astype(int))
        y_t = pd.Series(y_decoded, index=y_pd.index)
        y_t_ww = ww.DataColumn(y_t, logical_type=label_encoder._y_logical_type)
        return X, y_t_ww

    def fit_transform(self, X, y, dependent_components=None):
        if dependent_components is None:
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        return self.fit(X, y).transform(X, y, dependent_components=dependent_components)
