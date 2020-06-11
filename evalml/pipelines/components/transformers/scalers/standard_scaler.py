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
