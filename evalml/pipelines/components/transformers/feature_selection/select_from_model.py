import numpy as np
from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from evalml.pipelines.components import ComponentTypes
# from .feature_selector import FeatureSelector
from evalml.pipelines.components.transformers import Transformer


class SelectFromModel(Transformer):
    """Selects top features based on importance weights"""
    hyperparameter_ranges = {
        "percent_features": Real(.01, 1),
        "threshold": ['mean', -np.inf]
    }

    def __init__(self, estimator, number_features, percent_features=0.5, threshold=-np.inf):
        name = 'Select From Model'
        component_type = ComponentTypes.FEATURE_SELECTION
        max_features = max(1, int(percent_features * number_features))
        parameters = {"percent_features": percent_features,
                      "threshold": threshold}
        feature_selection = SkSelect(
            estimator=estimator,
            max_features=max_features,
            threshold=threshold
        )
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=feature_selection,
                         needs_fitting=True,
                         random_state=0)
