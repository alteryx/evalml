import pandas as pd
from sklearn.preprocessing import LabelEncoder as SkLabelEncoder

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import _convert_woodwork_types_wrapper


class LabelEncoder(Transformer):
    """Encodes target from string (or other dtype) to integers ranging from 0 to n-1."""
    name = 'Label Encoder'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        super().__init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        self._encoder = SkLabelEncoder()
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        self._encoder.fit(y_pd)
        return self

    def transform(self, X, y=None):
        y_pd = _convert_woodwork_types_wrapper(y.to_series())
        try:
            return X, pd.Series(self._encoder.transform(y_pd), index=y_pd.index, name=y.name)
        except ValueError as e:
            raise ValueError(str(e))

    def fit_transform(self, X, y):
        import pdb; pdb.set_trace()
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
        if dependent_components is None:
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        return X, self._encoder.inverse_transform(y.astype(int))

    def fit_transform(self, X, y, dependent_components=None):
        if dependent_components is None:
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        return self.fit(X, y).transform(X, y, dependent_components=dependent_components)
