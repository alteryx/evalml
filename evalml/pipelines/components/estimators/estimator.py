
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
        return self._component_obj.predict(X)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """
        return self._component_obj.predict_proba(X)
