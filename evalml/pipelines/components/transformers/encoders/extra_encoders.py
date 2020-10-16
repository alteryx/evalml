from evalml.pipelines.components.transformers.transformer import Transformer
import category_encoders as ce


class BinaryEncoder(Transformer):
    name = "Binary Encoder"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0, **kwargs):
        super().__init__({}, component_obj=None, random_state=random_state, **kwargs)
        self._encoder = None

    def fit(self, X, y=None):
        cat_cols = list(X.select_dtypes(["object", "category"]).columns)
        self._encoder = ce.BinaryEncoder(cat_cols)
        self._encoder.fit(X)

    def transform(self, X, y=None):
        return self._encoder.transform(X)


class SumEncoder(Transformer):
    name = "Sum Encoder"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0, **kwargs):
        super().__init__({}, component_obj=None, random_state=random_state, **kwargs)
        self._encoder = None

    def fit(self, X, y=None):
        cat_cols = list(X.select_dtypes(["object", "category"]).columns)
        self._encoder = ce.SumEncoder(cat_cols)
        self._encoder.fit(X)

    def transform(self, X, y=None):
        return self._encoder.transform(X)


class OrdinalEncoder(Transformer):
    name = "Ordinal Encoder"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0, **kwargs):
        super().__init__({}, component_obj=None, random_state=random_state, **kwargs)
        self._encoder = None

    def fit(self, X, y=None):
        cat_cols = list(X.select_dtypes(["object", "category"]).columns)
        self._encoder = ce.SumEncoder(cat_cols)
        self._encoder.fit(X)

    def transform(self, X, y=None):
        return self._encoder.transform(X)

