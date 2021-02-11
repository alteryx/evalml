import pandas as pd

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.transformers import Transformer
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    _retain_custom_types_and_initalize_woodwork,
    infer_feature_types
)


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_names(self):
        """Get names of selected features.

        Returns:
            list[str]: List of the names of features selected
        """
        selected_masks = self._component_obj.get_support()
        return [feature_name for (selected, feature_name) in zip(selected_masks, self.input_feature_names) if selected]

    def transform(self, X, y=None):
        """Transforms input data by selecting features. If the component_obj does not have a transform method, will raise an MethodPropertyNotFoundError exception.

        Arguments:
            X (ww.DataTable, pd.DataFrame): Data to transform.
            y (ww.DataColumn, pd.Series, optional): Target data. Ignored.

        Returns:
            ww.DataTable: Transformed X
        """
        X_ww = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X_ww.to_dataframe())
        self.input_feature_names = list(X.columns.values)

        try:
            X_t = self._component_obj.transform(X)
        except AttributeError:
            raise MethodPropertyNotFoundError("Feature selector requires a transform method or a component_obj that implements transform")

        X_dtypes = X.dtypes.to_dict()
        selected_col_names = self.get_names()
        col_types = {key: X_dtypes[key] for key in selected_col_names}
        features = pd.DataFrame(X_t, columns=selected_col_names, index=X.index).astype(col_types)
        return _retain_custom_types_and_initalize_woodwork(X_ww, features)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
