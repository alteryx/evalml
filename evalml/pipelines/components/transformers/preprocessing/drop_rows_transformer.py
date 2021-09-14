"""Transformer to drop rows specified by row indices."""
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class DropRowsTransformer(Transformer):
    """Transformer to drop rows specified by row indices.

    Args:
        indices_to_drop (list): List of indices to drop in the input data. Defaults to None.
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop Rows Transformer"
    modifies_target = True
    training_only = True
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, indices_to_drop=None, random_seed=0):
        if indices_to_drop is not None and len(set(indices_to_drop)) != len(
            indices_to_drop
        ):
            raise ValueError("All input indices must be unique.")
        self.indices_to_drop = indices_to_drop
        super().__init__(parameters=None, component_obj=None, random_seed=random_seed)

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self

        Raises:
            ValueError: If indices to drop do not exist in input features or target.
        """
        X_t = infer_feature_types(X)
        y_t = infer_feature_types(y) if y is not None else None
        if self.indices_to_drop is not None:
            indices_to_drop_set = set(self.indices_to_drop)
            missing_X_indices = indices_to_drop_set.difference(set(X_t.index))
            missing_y_indices = (
                indices_to_drop_set.difference(set(y_t.index))
                if y_t is not None
                else None
            )
            if len(missing_X_indices):
                raise ValueError(
                    "Indices [{}] do not exist in input features".format(
                        list(missing_X_indices)
                    )
                )
            elif y_t is not None and len(missing_y_indices):
                raise ValueError(
                    "Indices [{}] do not exist in input target".format(
                        list(missing_y_indices)
                    )
                )
        return self

    def transform(self, X, y=None):
        """Transforms data using fitted component.

        Args:
            X (pd.DataFrame): Features.
            y (pd.Series, optional): Target data.

        Returns:
            (pd.DataFrame, pd.Series): Data with row indices dropped.
        """
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
