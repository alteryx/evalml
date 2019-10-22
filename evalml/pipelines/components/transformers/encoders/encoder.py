
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class Encoder(Transformer):

    def get_feature_names(self):
        return self._component_obj.feature_names