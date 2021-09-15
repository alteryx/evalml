"""Component that selects top features based on importance weights."""
import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import infer_feature_types


class FeatureSelector(Transformer):
    """Selects top features based on importance weights.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def get_names(self):
        """Get names of selected features.

        Returns:
            list[str]: List of the names of features selected.
        """
        selected_masks = self._component_obj.get_support()
        return [
            feature_name
            for (selected, feature_name) in zip(
                selected_masks, self.input_feature_names
            )
            if selected
        ]

    def transform(self, X, y=None):
        """Transforms input data by selecting features. If the component_obj does not have a transform method, will raise an MethodPropertyNotFoundError exception.

        Args:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Target data. Ignored.

        Returns:
            pd.DataFrame: Transformed X

        Raises:
            MethodPropertyNotFoundError: If feature selector does not have a transform method or a component_obj that implements transform
        """
        X_ww = infer_feature_types(X)
        self.input_feature_names = list(X_ww.columns.values)

        try:
            X_t = self._component_obj.transform(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Feature selector requires a transform method or a component_obj that implements transform"
            )

        selected_col_names = self.get_names()
        features = pd.DataFrame(X_t, columns=selected_col_names, index=X_ww.index)
        features.ww.init(schema=X_ww.ww.schema._get_subset_schema(selected_col_names))
        return features

    def fit_transform(self, X, y=None):
        """Fit and transform data using the feature selector.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series, optional): The target training data of length [n_samples].

        Returns:
            pd.DataFrame: Transformed data.
        """
        return self.fit(X, y).transform(X, y)
