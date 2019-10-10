from sklearn.preprocessing import StandardScaler as SkScaler

from .component_types import ComponentTypes
from .transformer import Transformer


class StandardScaler(Transformer):
    """Standardize features: removes mean and scales to unit variance"""

    def __init__(self):
        self.name = 'Standard Scaler'
        self.component_type = ComponentTypes.SCALER
        self.hyperparameters = {}
        self.parameters = {}
        scaler = SkScaler()
        super().__init__(name=self.name, component_type=self.component_type, hyperparameters=self.hyperparameters, needs_fitting=True, component_obj=scaler)
