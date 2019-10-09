from sklearn.preprocessing import StandardScaler as SkScaler

from .component_types import ComponentTypes
from .transformer import Transformer


class StandardScaler(Transformer):
    "Standardize features: removes meand and scales to unit variance"
    def __init__(self):
        name = 'Standard Scaler'
        component_type = ComponentTypes.SCALER
        hyperparameters = None

        scaler = SkScaler()
        super().__init__(name=name, component_type=component_type, hyperparameters=hyperparameters, needs_fitting=True, component_obj=scaler)
