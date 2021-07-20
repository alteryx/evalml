from abc import abstractmethod

import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components import ComponentBase
from evalml.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types,
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

    Arguments:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    model_family = ModelFamily.NONE
    """ModelFamily.NONE"""
    modifies_features = True
    modifies_target = False

    def transform(self, X, y=None):
        """Transforms data X.

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Target data.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        try:
            X_t = self._component_obj.transform(X, y)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Transformer requires a transform method or a component_obj that implements transform"
            )
        X_t_df = pd.DataFrame(X_t, columns=X_ww.columns, index=X_ww.index)
        return _retain_custom_types_and_initalize_woodwork(
            X_ww.ww.logical_types, X_t_df
        )

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Target data

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        if y is not None:
            y_ww = infer_feature_types(y)
        try:
            X_t = self._component_obj.fit_transform(X_ww, y_ww)
            return _retain_custom_types_and_initalize_woodwork(
                X_ww.ww.logical_types, X_t
            )
        except AttributeError:
            try:
                return self.fit(X, y).transform(X, y)
            except MethodPropertyNotFoundError as e:
                raise e

    def _get_feature_provenance(self):
        return {}


class TargetTransformer(Transformer):
    """A component that transforms the target."""

    modifies_features = False
    modifies_target = True

    @abstractmethod
    def inverse_transform(self, y):
        """Inverts the transformation done by the transform method.

         Arguments:
            y (pd.Series): Target transformed by this component.

        Returns:
            pd.Seriesø: Target without the transformation.
        """
