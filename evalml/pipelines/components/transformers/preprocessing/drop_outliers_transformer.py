"""Transformer to drop rows specified by row indices."""
from evalml.data_checks.outliers_data_check import OutliersDataCheck
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class DropOutliersTransformer(Transformer):
    """Transformer to drop outliers.

    Args:
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop Outliers Transformer"
    modifies_target = True
    training_only = True
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, random_seed=0):
        parameters = {}
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

    def fit(self, X, y=None):
        X_t = infer_feature_types(X)
        self.outlier_rows = OutliersDataCheck.get_outlier_rows(X_t)
        return self

    def transform(self, X, y=None):
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None

        all_rows_with_indices_set = set()
        for row_indices in self.outlier_rows.values():
            all_rows_with_indices_set.update(row_indices)

        all_rows_with_indices = list(all_rows_with_indices_set)
        all_rows_with_indices.sort()
        self.outlier_indices = all_rows_with_indices

        X_t = X_t.drop(self.all_rows_with_indices, axis=0)
        X_t.ww.init()

        if y_t is not None:
            y_t = y_t.drop(self.all_rows_with_indices)
            y_t.ww.init()
        return X_t, y_t
