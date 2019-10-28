from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        indices = self._component_obj.get_support(indices=True)
        return indices


    def get_names(self, all_feature_names):
        """Get names of selected features.

        Args:
            all_feature_names: feature names

        Returns:
            list of the names of features selected
        """
        indices = self.get_indices()
        return list(map(lambda i: all_feature_names[i], indices))
