"""Transformer to drop rows specified by row indices."""
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import drop_rows_with_nans, infer_feature_types


class DropNaNRows(Transformer):
    """Transformer to drop rows specified by row indices.

    Args:
        first_rows_to_drop (list): Only checks first N rows when dropping NaN values. If None, checks all rows. Defaults to None.
        random_seed (int): Seed for the random number generator. Is not used by this component. Defaults to 0.
    """

    name = "Drop NaN Rows Transformer"
    modifies_target = True
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, first_rows_to_drop=None, random_seed=0):
        # self.X_t_indicies = None
        # self.y_t_indicies = None
        self.first_rows_to_drop = first_rows_to_drop
        parameters = {"first_rows_to_drop": self.first_rows_to_drop}
        super().__init__(
            parameters=parameters, component_obj=None, random_seed=random_seed
        )

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

        X_t, y_t = drop_rows_with_nans(X_t, y_t)
        X_t.ww.init()
        if y_t is not None:
            y_t.ww.init()
        return X_t, y_t
