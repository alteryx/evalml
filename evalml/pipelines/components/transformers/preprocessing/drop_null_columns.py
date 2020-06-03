import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DropNullColumns(Transformer):
    """Transformer to drop features whose percentage of NaN values exceeds a specified threshold"""
    name = "Drop Null Columns Transformer"
    hyperparameter_ranges = {}

    def __init__(self, pct_null_threshold=1.0, random_state=0):
        """Initalizes an transformer to drop features whose percentage of NaN values exceeds a specified threshold.

        Arguments:
            pct_null_threshold(float): The percentage of NaN values in an input feature to drop. Defaults to 0.95.
        """
        if pct_null_threshold < 0 or pct_null_threshold > 1:
            raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")
        parameters = {"pct_null_threshold": pct_null_threshold}
        self.cols_to_drop = None
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=random_state)

    def fit(self, X, y=None):
        pct_null_threshold = self.parameters["pct_null_threshold"]
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        percent_null = (X.isnull().mean()).to_dict()
        if pct_null_threshold == 0.0:
            null_cols = {key: value for key, value in percent_null.items() if value > 0.0}
        else:
            null_cols = {key: value for key, value in percent_null.items() if value >= self.pct_null_threshold}
        self.cols_to_drop = null_cols.keys()
        return self

    def transform(self, X, y=None):
        """Transforms data X by dropping columns that exceed the threshold of null values.
        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Targets
        Returns:
            pd.DataFrame: Transformed X
        """
        if self.cols_to_drop is None:
            raise RuntimeError("You must fit Drop Null Columns transformer before calling transform!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop, axis=1)
