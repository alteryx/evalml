import pandas as pd

from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        """Get integer index of features selected

        Returns:
            list: list of indices
        """
        indices = self._component_obj.get_support(indices=True)
        return indices

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
            y (pd.Series, optional): Input Labels
        Returns:
            pd.DataFrame: Transformed X
        """
        if isinstance(X, pd.DataFrame):
            self.input_feature_names = list(X.columns.values)
        else:
            self.input_feature_names = range(len(X.shape[1]))

        try:
            X_t = self._component_obj.transform(X)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_dtypes = X.dtypes.to_dict()
                selected_col_names = self.get_names()
                col_types = {key: X_dtypes[key] for key in selected_col_names}
                X_t = pd.DataFrame(X_t, columns=selected_col_names, index=X.index).astype(col_types)
            return X_t
        except AttributeError:
            raise RuntimeError("Transformer requires a transform method or a component_obj that implements transform")

    def fit_transform(self, X, y=None):
        """Fits feature selector on data X then transforms X by selecting features

        Arguments:
            X (pd.DataFrame): Data to fit and transform
            y (pd.Series): Labels to fit and transform
        Returns:
            pd.DataFrame: Transformed X
        """
        if isinstance(X, pd.DataFrame):
            self.input_feature_names = list(X.columns.values)
        else:
            self.input_feature_names = range(len(X.shape[1]))

        try:
            X_t = self._component_obj.fit_transform(X, y)
            if not isinstance(X_t, pd.DataFrame) and isinstance(X, pd.DataFrame):
                X_dtypes = X.dtypes.to_dict()
                selected_col_names = self.get_names()
                col_types = {key: X_dtypes[key] for key in selected_col_names}
                X_t = pd.DataFrame(X_t, columns=selected_col_names, index=X.index).astype(col_types)
            return X_t
        except AttributeError:
            raise RuntimeError("Transformer requires a fit_transform method or a component_obj that implements fit_transform")
