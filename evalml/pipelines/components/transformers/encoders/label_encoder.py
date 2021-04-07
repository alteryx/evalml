from evalml.pipelines.components.transformers import Transformer


class LabelEncoder(Transformer):
    """Encodes target from string (or other dtype) to integers ranging from 0 to n-1."""
    name = 'Label Encoder'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        super().init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        self._encoder = LabelEncoder()
        self._encoder.fit(y)

    def transform(self, X, y=None):
        try:
            return X, pd.Series(self._encoder.transform(y), index=y.index, name=y.name)
        except ValueError as e:
            raise ValueError(str(e))


class LabelDecoder(Transformer):
    """Decodes target from integers from 0 to n-1 to string or other original dtype"""
    name = 'Label Decoder'
    hyperparameter_ranges = {}

    def __init__(self, random_seed=0):
        super().init__(parameters={}, component_obj=None, random_seed=random_seed)

    def fit(self, X, y):
        """No-op"""

    def transform(self, X, y=None, dependent_components=None):
        if dependent_components is None:
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        return X, self._encoder.inverse_transform(y.astype(int))

    def fit_transform(self, X, y, dependent_components=None):
        if dependent_components is None:
            raise Exception('A reference to a previously-used label encoder component must be provided in order to decode labels.')
        return self.fit(X, y).transform(X, y, dependent_components=dependent_components)
