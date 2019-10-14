from evalml.pipelines.components import ComponentBase


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data.
    These components are used before an estimator.
    """

    def __init__(self, name, component_type, parameters={}, needs_fitting=False, component_obj=None):
        super().__init__(name=name, component_type=component_type, parameters=parameters, needs_fitting=needs_fitting, component_obj=component_obj)

    def transform(self, X):
        """Transforms data X

        Arguments:
            X (DataFrame): Data to transform

        Returns:
            DataFrame: Transformed X
        """
        try:
            return self._component_obj.transform(X)
        except AttributeError:
            raise RuntimeError("Transformer requires a transform method or a component_obj that implements transform")


    def fit_transform(self, X):
        """Fits on X and transforms X

        Arguments:
            X (DataFrame): Data to fit and transform

        Returns:
            DataFrame: Transformed X
        """
        try:
            return self._component_obj.fit_transform(X)
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
