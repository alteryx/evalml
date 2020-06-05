from sklearn.base import clone as sk_clone
from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.pipelines.components.transformers import Transformer


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""
    name = "Standard Scaler"
    hyperparameter_ranges = {}

    def __init__(self, random_state=0):
        parameters = {}
        scaler = SkScaler()
        super().__init__(parameters=parameters,
                         component_obj=scaler,
                         random_state=random_state)

    def clone(self):
        cloned_obj = StandardScaler()
        cloned_scaler = sk_clone(self._component_obj)
        cloned_obj._component_obj = cloned_scaler
        return cloned_obj
