import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types,
)


class FeatureSelector(Transformer):
    """
    Selects top features based on importance weights.

    Arguments:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    def get_names(self):
        """Get names of selected features.

        Returns:
            list[str]: List of the names of features selected
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

        Arguments:
            X (pd.DataFrame): Data to transform.
            y (pd.Series, optional): Target data. Ignored.

        Returns:
            pd.DataFrame: Transformed X
        """
        X_ww = infer_feature_types(X)
        self.input_feature_names = list(X_ww.columns.values)

        try:
            X_t = self._component_obj.transform(X)
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Feature selector requires a transform method or a component_obj that implements transform"
            )

        X_dtypes = X_ww.dtypes.to_dict()
        selected_col_names = self.get_names()
        col_types = {key: X_dtypes[key] for key in selected_col_names}
        features = pd.DataFrame(
            X_t, columns=selected_col_names, index=X_ww.index
        ).astype(col_types)
        return _retain_custom_types_and_initalize_woodwork(
            X_ww.ww.logical_types, features
        )

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
