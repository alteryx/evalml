from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class DropRowsTransformer(Transformer):
    """Transformer to drop rows specified by row indices.


    Arguments:
        indices_to_drop (list): List of indices to drop in the input data.
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop Rows Transformer"
    modifies_target = True
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, indices_to_drop=None, random_seed=0):
        self.indices_to_drop = indices_to_drop
        super().__init__(parameters=None, component_obj=None, random_seed=0)

    def fit(self, X, y=None):
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None

        if self.indices_to_drop is not None:
            indices_to_drop_set = set(self.indices_to_drop)
            if not indices_to_drop_set.issubset(X_t.index):
                raise ValueError("Index does not exist in input features.")
            elif y_t is not None and not indices_to_drop_set.issubset(y_t.index):
                raise ValueError("Index does not exist in input target.")
        return self

    def transform(self, X, y=None):
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None
        if self.indices_to_drop is None or len(self.indices_to_drop) == 0:
            return X_t, y_t

        X_t = X_t.drop(self.indices_to_drop, axis=0)
        X_t.ww.init()

        if y_t is not None:
            y_t = y_t.drop(self.indices_to_drop)
            y_t.ww.init()
        return X_t, y_t
