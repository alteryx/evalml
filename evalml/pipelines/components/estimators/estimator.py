from evalml.pipelines.components import ComponentBase


class Estimator(ComponentBase):
    """A component that fits and predicts given data"""

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

    @property
    def feature_importances(self):
        try:
            return self._component_obj.feature_importances_
        except AttributeError:
            raise RuntimeError("Estimator requires a feature_importances property or a component_obj that implements feature_importances_")
