import pandas as pd

from evalml.pipelines.components import ComponentBase


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data.
    These components are used before an estimator.
    """

    def transform(self, X, y=None):
        """Transforms data X

        Arguments:
            X (pd.DataFrame): Data to transform

        Returns:
            pd.DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.transform(X)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)
            return X_t
        except AttributeError:
            raise RuntimeError("Transformer requires a transform method or a component_obj that implements transform")

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform

        Returns:
            pd.DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.fit_transform(X, y)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)
            return X_t
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
