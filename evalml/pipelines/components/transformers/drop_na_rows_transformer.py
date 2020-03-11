import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class DropNaRowsTransformer(Transformer):
    """Drops rows when a given column has a null value."""

    name = 'Drop NA Row Transformer'
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
        # na_rows = y.isna()
        return X.dropna(axis=0)

    def fit_transform(self, X, y=None):
        """Fits

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        # X_t = self._component_obj.fit_transform(X, y)
        # if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
        #     # skLearn's SimpleImputer loses track of column type, so we need to restore
        #     X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index).astype(X.dtypes.to_dict())
        # return X_t
        return self.transform(X, y)
