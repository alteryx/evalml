import numpy as np
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class FeatureSelector(Transformer):
    """Selects top features based on importance weights"""

    def get_indices(self):
        indices = self._component_obj.get_support(indices=True)
        return indices