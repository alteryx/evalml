from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class DropRowsTransformer(Transformer):
    """Transformer to drop rows specified by row indices."""

    name = "Drop Rows Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, indices_to_drop=None):
        super().__init__(parameters=None, component_obj=None, random_seed=0)
        self.indices_to_drop = indices_to_drop

    def fit(self, X, y=None):
        X_t = infer_feature_types(X)
        # check if all indices exist. If no, error
        if self.indices_to_drop is not None:
            for index in self.indices_to_drop:
                if index not in X_t.index:
                    raise ValueError("Index does not exist in input DataFrame")
        return self

    def transform(self, X, y=None):
        X_t = infer_feature_types(X)
        X_t
        if y is not None:
            y_t = infer_feature_types(y)
        # check if y is None
        if len(self.indices_to_drop) == 0:
            return X_t
        rows_dropped = X_t.drop(self.indices_to_drop, axis=0)
        rows_dropped.ww.init()
        return rows_dropped
