from sklearn.preprocessing import StandardScaler as SkScaler

from .transformer import Transformer

from evalml.pipelines.components import ComponentTypes


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""
    hyperparameters = {}

    def __init__(self):
        self.name = "Standard Scaler"
        self.component_type = ComponentTypes.SCALER
        self.parameters = {}
        scaler = SkScaler()
        super().__init__(name=self.name, component_type=self.component_type, needs_fitting=True, component_obj=scaler)
