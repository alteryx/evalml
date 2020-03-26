from abc import abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase


class Estimator(ComponentBase):
    """A component that fits and predicts given data.

    To implement a new Estimator, define your own class which is a subclass of Estimator. Define
    a name for the estimator, and a list of acceptable ranges for hyperparameters. Then define
    an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments, and ends with a call to `super().__init__()`. You may
    also override the `fit`, `predict`/`predict_proba` and other methods in this class if appropriate.

    Check out the definitions of any Estimator components to see some examples.
    """

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        return NotImplementedError("This component must have `supported_problem_types` as a class variable.")

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.Series : estimated labels
        """
        try:
            return self._component_obj.predict(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a predict method or a component_obj that implements predict")

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (pd.DataFrame) : features

        Returns:
            pd.DataFrame : probability estimates
        """
        try:
            return self._component_obj.predict_proba(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a predict_proba method or a component_obj that implements predict_proba")

    @property
    def feature_importances(self):
        """Returns feature importances.

        Returns:
            list(float) : importance associated with each feature
        """
        try:
            return self._component_obj.feature_importances_
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a feature_importances property or a component_obj that implements feature_importances_")
