"""Transformer to drop features whose percentage of NaN values exceeds a specified threshold."""
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class DropNullColumns(Transformer):
    """Transformer to drop features whose percentage of NaN values exceeds a specified threshold.

    Args:
        pct_null_threshold(float): The percentage of NaN values in an input feature to drop.
            Must be a value between [0, 1] inclusive. If equal to 0.0, will drop columns with any null values.
            If equal to 1.0, will drop columns with all null values. Defaults to 0.95.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Drop Null Columns Transformer"
    hyperparameter_ranges = {}
    """{}"""

    def __init__(self, pct_null_threshold=1.0, random_seed=0, **kwargs):
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError(
                "pct_null_threshold must be a float between 0 and 1, inclusive.",
            )
        parameters = {"pct_null_threshold": pct_null_threshold}
        parameters.update(kwargs)

        self._cols_to_drop = None
        super().__init__(
            parameters=parameters,
            component_obj=None,
            random_seed=random_seed,
        )

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            self
        """
        pct_null_threshold = self.parameters["pct_null_threshold"]
        X_t = infer_feature_types(X)
        percent_null = X_t.isnull().mean()
        if pct_null_threshold == 0.0:
            null_cols = percent_null[percent_null > 0]
        else:
            null_cols = percent_null[percent_null >= pct_null_threshold]
        self._cols_to_drop = list(null_cols.index)
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns that exceed the threshold of null values.

        Args:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = infer_feature_types(X)
        if len(self._cols_to_drop) == 0:
            return X_t
        return X_t.ww.drop(self._cols_to_drop)
