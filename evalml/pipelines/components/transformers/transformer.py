from evalml.pipelines.components import ComponentBase


class Transformer(ComponentBase):
    """A component that may or may not need fitting that transforms data.
    These components are used before an estimator.
    """

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

    def fit_transform(self, X, y=None):
        """Fits on X and transforms X

        Arguments:
            X (DataFrame): Data to fit and transform

        Returns:
            DataFrame: Transformed X
        """
        try:
            return self._component_obj.fit_transform(X, y)
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
