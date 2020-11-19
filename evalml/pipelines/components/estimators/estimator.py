from abc import abstractmethod

import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)


class Estimator(ComponentBase):
    """A component that fits and predicts given data.

    To implement a new Transformer, define your own class which is a subclass of Transformer, including
    a name and a list of acceptable ranges for any parameters to be tuned during the automl search (hyperparameters).
    Define an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments and calls `super().__init__()` with a parameters dict. You may also override the
    `fit`, `transform`, `fit_transform` and other methods in this class if appropriate.

    To see some examples, check out the definitions of any Estimator component.
    """

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        """Problem types this estimator supports"""

    def predict(self, X):
        """Make predictions using selected features.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Data of shape [n_samples, n_features]

        Returns:
            pd.Series: Predicted values
        """
        try:
            X = _convert_to_woodwork_structure(X)
            X = _convert_woodwork_types_wrapper(X.to_dataframe())
            predictions = self._component_obj.predict(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a predict method or a component_obj that implements predict")
        if not isinstance(predictions, pd.Series):
            predictions = pd.Series(predictions)
        return predictions

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Arguments:
            X (ww.DataTable, pd.DataFrame, or np.ndarray): Features

        Returns:
            pd.DataFrame: Probability estimates
        """
        try:
            X = _convert_to_woodwork_structure(X)
            X = _convert_woodwork_types_wrapper(X.to_dataframe())
            pred_proba = self._component_obj.predict_proba(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a predict_proba method or a component_obj that implements predict_proba")
        if not isinstance(pred_proba, pd.DataFrame):
            pred_proba = pd.DataFrame(pred_proba)
        return pred_proba

    @property
    def feature_importance(self):
        """Returns importance associated with each feature.

        Returns:
            list(float): Importance associated with each feature
        """
        try:
            return self._component_obj.feature_importances_
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a feature_importance property or a component_obj that implements feature_importances_")

    def __eq__(self, other):
        return super().__eq__(other) and self.supported_problem_types == other.supported_problem_types
