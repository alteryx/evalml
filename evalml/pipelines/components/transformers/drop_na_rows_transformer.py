import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DropNaNRowsTransformer(Transformer):
    """Transformer that drop any rows with a null value."""

    name = 'Drop NaN Row Transformer'
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        super().__init__(parameters=parameters,
                         component_obj=None,
                         random_state=0)

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        return self

    def transform(self, X, y=None):
        """Transforms data X

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # drop rows where corresponding y is NaN
        null_indices = y.index[y.apply(np.isnan)]
        X_t = X.drop(index=null_indices)

        # drop any rows with NaN
        X_t = X_t.dropna(axis=0)
        return X_t

    def fit_transform(self, X, y=None):
        """Fits

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        return self.transform(X, y)
