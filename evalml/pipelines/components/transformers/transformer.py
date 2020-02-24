import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data.
    These components are used before an estimator.
    """

    def transform(self, X, y=None):
        """Transforms data X

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.transform(X)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)
            return X_t
        except AttributeError:
            raise MethodPropertyNotFoundError("Transformer requires a transform method or a component_obj that implements transform")

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd. DataFrame): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.fit_transform(X, y)
        except AttributeError:
            try:
                self.fit(X, y)
                X_t = self.transform(X, y)
            except MethodPropertyNotFoundError as e:
                raise e

        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            X_t = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return X_t
