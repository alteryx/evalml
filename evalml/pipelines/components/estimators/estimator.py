
from evalml.pipelines.components import ComponentBase


class Estimator(ComponentBase):
    """A component that fits and predicts given data"""

    def __init__(self, name, component_type, parameters={}, needs_fitting=False, component_obj=None, random_state=0):
        super().__init__(name=name, component_type=component_type, parameters=parameters, needs_fitting=needs_fitting,
                         component_obj=component_obj, random_state=random_state)

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        try:
            return self._component_obj.predict(X)
        except AttributeError:
            raise RuntimeError("Estimator requires a predict method or a component_obj that implements predict")

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """
        try:
            return self._component_obj.predict_proba(X)
        except AttributeError:
            raise RuntimeError("Estimator requires a predict_proba method or a component_obj that implements predict_proba")
