from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""
    name = "Standard Scaler"
    component_type = ComponentTypes.SCALER
    _needs_fitting = True
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        scaler = SkScaler()
        super().__init__(parameters=parameters,
                         component_obj=scaler,
                         random_state=0)
