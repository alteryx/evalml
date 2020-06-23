
from evalml.pipelines.components.transformers import Transformer


class CategoricalEncoder(Transformer):
    """Base class for encoding num-numeric data."""

    def get_feature_names(self):
        """Returns names of transformed and added columns

        Returns:
            list: list of feature names not including dropped features
        """
        return self._component_obj.feature_names
