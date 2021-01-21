import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


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
            X (ww.DataTable, pd.DataFrame): Data to transform
            y (ww.DataColumn, pd.Series, optional): Target data

        Returns:
            ww.DataTable: Transformed X
        """
        try:
            X = _convert_to_woodwork_structure(X)
            X = _convert_woodwork_types_wrapper(X.to_dataframe())
            if y is not None:
                y = _convert_to_woodwork_structure(y)
                y = _convert_woodwork_types_wrapper(y.to_series())
            # category treated differently from object...
            X_t = self._component_obj.transform(X, y)
        except AttributeError:
            raise MethodPropertyNotFoundError("Transformer requires a transform method or a component_obj that implements transform")

        X_t_df = pd.DataFrame(X_t, columns=X.columns, index=X.index)
        return _convert_to_woodwork_structure(X_t_df)

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to fit and transform
            y (ww.DataColumn, pd.Series): Target data

        Returns:
            ww.DataTable: Transformed X
        """
        try:
            X2 = _convert_to_woodwork_structure(X)
            y2 = _convert_to_woodwork_structure(y)
            X2 = _convert_woodwork_types_wrapper(X2.to_dataframe())
            y2 = _convert_woodwork_types_wrapper(y2.to_series())
            X_t = self._component_obj.fit_transform(X2, y2)
        except AttributeError:
            try:
                self.fit(X, y)
                X_t = self.transform(X, y)
            except MethodPropertyNotFoundError as e:
                raise e
        return _convert_to_woodwork_structure(X_t)
