from evalml.pipelines.components.transformers import Transformer
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class DropNullColumns(Transformer):
    """Transformer to drop features whose percentage of NaN values exceeds a specified threshold"""
    name = "Drop Null Columns Transformer"
    hyperparameter_ranges = {}

    def __init__(self, pct_null_threshold=1.0, random_state=0, **kwargs):
        """Initalizes an transformer to drop features whose percentage of NaN values exceeds a specified threshold.

        Arguments:
            pct_null_threshold(float): The percentage of NaN values in an input feature to drop.
                Must be a value between [0, 1] inclusive. If equal to 0.0, will drop columns with any null values.
                If equal to 1.0, will drop columns with all null values. Defaults to 0.95.
        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        parameters = {"pct_null_threshold": pct_null_threshold}
        parameters.update(kwargs)

        self._cols_to_drop = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        pct_null_threshold = self.parameters["pct_null_threshold"]
        X_t = _convert_to_woodwork_structure(X)
        X_t = _convert_woodwork_types_wrapper(X_t.to_dataframe())
        percent_null = X_t.isnull().mean()
        if pct_null_threshold == 0.0:
            null_cols = percent_null[percent_null > 0]
        else:
            null_cols = percent_null[percent_null >= pct_null_threshold]
        self._cols_to_drop = list(null_cols.index)
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns that exceed the threshold of null values.

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_t = _convert_to_woodwork_structure(X)
        X_t = _convert_woodwork_types_wrapper(X_t.to_dataframe())
        return X_t.drop(columns=self._cols_to_drop, axis=1)
