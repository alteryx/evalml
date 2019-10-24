from sklearn.preprocessing import StandardScaler as SkScaler

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.transformers import Transformer


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""
    name = "Standard Scaler"
    component_type = ComponentTypes.SCALER
    hyperparameter_ranges = {}

    def __init__(self):
        parameters = {}
        scaler = SkScaler()
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=scaler,
                         needs_fitting=True,
                         random_state=0)
