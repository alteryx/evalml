from abc import abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components import ComponentBase


class Estimator(ComponentBase):
    """A component that fits and predicts given data"""

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        return NotImplementedError("This component must have `name` as a class variable.")

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
        try:
            return self._component_obj.feature_importances_
        except AttributeError:
            raise MethodPropertyNotFoundError("Estimator requires a feature_importances property or a component_obj that implements feature_importances_")
