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
        """Computes the delayed features for all features in X and y.

        For each feature in X, it will add a column to the output dataframe for each
        delay in the (inclusive) range [1, max_delay]. The values would correspond to the
        value of the feature 'delay' time-steps in the past.

        If y is not None, it will also compute the delayed values for the target variable.

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Targets.

        Returns:
            pd.DataFrame: Transformed X.
        """
        if not isinstance(X, pd.DataFrame):
            # The user only passed in the target as a Series
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

        # Handle cases where the target was passed in
        if y is not None:
            X = X.assign(**{f"target_delay_{t}": y.shift(t)
                            for t in range(self.max_delay + 1)})

        return X
