import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        indices = self._component_obj.get_support(indices=True)
        return indices

    def get_names(self, X):
        """Get names of selected features.

        Args:
            X(pd.DataFrame): features

        Returns:
            list of the names of features selected
        """
        indices = self.get_indices()
        names = [X.columns[i] for i in indices]
        return list(names)

    def transform(self, X):
        """Transforms data X

        Arguments:
            X (DataFrame): Data to transform

        Returns:
            DataFrame: Transformed X
        """
        try:
            X_t = self._component_obj.transform(X)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_t = pd.DataFrame(X_t, index=X.index, columns=self.get_names(X))
            return X_t
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
            X_t = self._component_obj.fit_transform(X, y)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_t = pd.DataFrame(X_t, index=X.index, columns=self.get_names(X))
            return X_t
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
