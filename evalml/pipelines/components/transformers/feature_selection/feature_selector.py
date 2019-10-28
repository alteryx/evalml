from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        indices = self._component_obj.get_support(indices=True)
        return indices

    def get_names(self, X):
        # WIP
        indices = self.get_indices()
        return X.columns[indices].tolist()
