import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_names(self):
        """Get names of selected features.

        Returns:
            list of the names of features selected
        """
        selected_masks = self._component_obj.get_support()
        return [feature_name for (selected, feature_name) in zip(selected_masks, self.input_feature_names) if selected]

    def transform(self, X, y=None):
        """Transforms data X by selecting features

        Arguments:
            X (pd.DataFrame): Data to transform
            y (pd.Series, optional): Target data

        Returns:
            pd.DataFrame: Transformed X
        """
        if isinstance(X, pd.DataFrame):
            self.input_feature_names = list(X.columns.values)
        else:
            self.input_feature_names = range(X.shape[1])

        try:
            X_t = self._component_obj.transform(X)
        except AttributeError:
            raise RuntimeError("Transformer requires a transform method or a component_obj that implements transform")
        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            X_dtypes = X.dtypes.to_dict()
            selected_col_names = self.get_names()
            col_types = {key: X_dtypes[key] for key in selected_col_names}
            return pd.DataFrame(X_t, columns=selected_col_names, index=X.index).astype(col_types)
        else:
            return pd.DataFrame(X_t)

    def fit_transform(self, X, y=None):
        """Fits feature selector on data X then transforms X by selecting features

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Target data

        Returns:
            pd.DataFrame: Transformed X
        """
        if isinstance(X, pd.DataFrame):
            self.input_feature_names = list(X.columns.values)
        else:
            self.input_feature_names = range(X.shape[1])

        try:
            X_t = self._component_obj.fit_transform(X, y)
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
        if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
            X_dtypes = X.dtypes.to_dict()
            selected_col_names = self.get_names()
            col_types = {key: X_dtypes[key] for key in selected_col_names}
            return pd.DataFrame(X_t, columns=selected_col_names, index=X.index).astype(col_types)
        else:
            return pd.DataFrame(X_t)
