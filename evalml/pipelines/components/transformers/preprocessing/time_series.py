import pandas as pd

from evalml.pipelines.components.transformers.transformer import Transformer


class DelayedFeaturesTransformer(Transformer):
    name = "Delayed Features Transformer"
    needs_time_series_parameters = True

    def __init__(self, max_lag=2, gap=1, random_state=0, **kwargs):

        parameters = {"max_lag": max_lag, "gap": gap}
        parameters.update(kwargs)
        super().__init__(parameters=parameters, random_state=random_state)
        self.max_lag = max_lag
        self.gap = gap

    def fit(self, X, y=None):
        """Fits the LaggedFeatureExtractor."""

    def transform(self, X, y=None):
        """Transforms the input data."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is not None and not isinstance(y, pd.Series):
            y = pd.Series(y)

        original_columns = X.columns
        X = X.assign(**{f"{col}_lag_{self.gap + t}": X[col].shift(t)
                        for t in range(self.max_lag + 1)
                        for col in X})
        X.drop(columns=original_columns, inplace=True)

        # Handle cases where the label was not passed
        if y is not None:
            X = X.assign(**{f"target_lag_{self.gap + t}": y.shift(t)
                            for t in range(self.max_lag + 1)})

        return X
