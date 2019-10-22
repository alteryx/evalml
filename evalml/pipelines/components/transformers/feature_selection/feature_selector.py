from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        indices = self._component_obj.get_support(indices=True)
        return indices
