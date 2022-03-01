"""Transformer to drop rows specified by row indices."""
from woodwork import init_series

from evalml.pipelines.components.transformers import Transformer
from evalml.utils import drop_rows_with_nans, infer_feature_types


class DropNaNRowsTransformer(Transformer):
    """Transformer to drop rows with NaN values.

    Args:
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop NaN Rows Transformer"
    modifies_target = True
    hyperparameter_ranges = {}
    """{}"""

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Transforms data using fitted component.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series, optional): Target data.

        Returns:
            (pd.DataFrame, pd.Series): Data with NaN rows dropped.
        """
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None

        X_t_schema = X_t.ww.schema
        if y_t is not None:
            y_t_logical = y_t.ww.logical_type
            y_t_semantic = y_t.ww.semantic_tags

        X_t, y_t = drop_rows_with_nans(X_t, y_t)
        X_t.ww.init_with_full_schema(X_t_schema)
        if y_t is not None:
            y_t = init_series(y_t, logical_type=y_t_logical, semantic_tags=y_t_semantic)
        return X_t, y_t
