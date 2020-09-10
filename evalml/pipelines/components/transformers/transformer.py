import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data.
    These components are used before an estimator.

    To implement a new Transformer, define your own class which is a subclass of Transformer, including
    a name and a list of acceptable ranges for any parameters to be tuned during the automl search (hyperparameters).
    Define an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments and calls `super().__init__()` with a parameters dict. You may also override the
    `fit`, `transform`, `fit_transform` and other methods in this class if appropriate.

    To see some examples, check out the definitions of any Transformer component.
    """

    model_family = ModelFamily.NONE

    def transform(self, X, y=None):
        """Transforms data X

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Target data
        Returns:
            pd.DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.transform(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Transformer requires a transform method or a component_obj that implements transform")
        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return pd.DataFrame(X_t)

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd. DataFrame): Target data
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
            return pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return pd.DataFrame(X_t)
