from abc import abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase


class Estimator(ComponentBase):
    """A component that fits and predicts given data

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
