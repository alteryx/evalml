import pandas as pd

from evalml.pipelines.components.transformers.transformer import Transformer


class DelayedFeaturesTransformer(Transformer):
    name = "Delayed Features Transformer"
    hyperparameter_ranges = {}
    needs_fitting = False

    def __init__(self, max_delay=2, random_state=0, **kwargs):

        parameters = {"max_delay": max_delay}
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_state=random_state)
        self.max_delay = max_delay

    def fit(self, X, y=None):
        """Fits the LaggedFeatureExtractor."""

    def transform(self, X, y=None):
        """Transforms the input data."""
        if not isinstance(X, pd.DataFrame):
            if y is None:
                X = pd.DataFrame(X, columns=["target"])
            else:
                X = pd.DataFrame(X)
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)

        original_columns = X.columns
        X = X.assign(**{f"{col}_delay_{t}": X[col].shift(t)
                        for t in range(self.max_delay + 1)
                        for col in X})
        X.drop(columns=original_columns, inplace=True)

        # Handle cases where the label was not passed
        if y is not None:
            X = X.assign(**{f"target_delay_{t}": y.shift(t)
                            for t in range(self.max_delay + 1)})

        return X
