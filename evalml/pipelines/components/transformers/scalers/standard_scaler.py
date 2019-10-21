from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""
    hyperparameter_ranges = {}

    def __init__(self):
        name = "Standard Scaler"
        component_type = ComponentTypes.SCALER
        parameters = {}
        scaler = SkScaler()
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=scaler,
                         needs_fitting=True,
                         random_state=0)
