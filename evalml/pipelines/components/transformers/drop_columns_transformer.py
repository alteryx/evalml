import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DropColumnsTransformer(Transformer):
    """Transformer to specified columns in input data."""
    name = "Drop Columns Transformer"
    hyperparameter_ranges = {}

    def __init__(self, columns=None, random_state=0):
        """Initalizes an transformer that drops specified columns in input data.
        Arguments:
            columns (list(string)): List of column names, used to determine which columns to drop.
        """
        if columns is None:
            columns = []
        parameters = {"columns": columns}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        cols = self.parameters["columns"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not set(cols).issubset(X.columns):
            raise ValueError("Columns to drop do not exist in input data")
        return X.drop(columns=cols, axis=1)

    def fit_transform(self, X, y=None):
        return self.transform(X, y)
