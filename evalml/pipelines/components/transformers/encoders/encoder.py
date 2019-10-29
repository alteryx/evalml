
from evalml.pipelines.components.transformers import Transformer


class CategoricalEncoder(Transformer):

    def get_feature_names(self):
        return self._component_obj.feature_names
