import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DropColumns(Transformer):
    """Transformer to specified columns in input data."""
    name = "Drop Columns Transformer"
    hyperparameter_ranges = {}

    def __init__(self, columns=None, random_state=0, **kwargs):
        """Initalizes an transformer that drops specified columns in input data.
        Arguments:
            columns (list(string)): List of column names, used to determine which columns to drop.
        """
        parameters = {"columns": columns}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def _check_input_for_columns(self, X):
        cols = self.parameters["columns"] or []
        missing_cols = set(cols) - set(X.columns)
        if len(missing_cols) > 0:
            raise ValueError("Columns {} not found in input data".format(', '.join(f"'{col_name}'" for col_name in missing_cols)))

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._check_input_for_columns(X)
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns.

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Targets

        Returns:
            pd.DataFrame: Transformed X
        """
        cols = self.parameters["columns"] or []
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self._check_input_for_columns(X)
        return X.drop(columns=cols, axis=1)
