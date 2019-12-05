import numpy as np
# from sklearn.feature_selection import SelectFromModel as SkSelect
from skopt.space import Real

from .feature_selector import FeatureSelector

from evalml.pipelines.components import ComponentTypes


class CatBoostClassifierSelectFromModel(FeatureSelector):
    """Selects top features based on importance weights using a CatBoost classifier"""
    name = 'CatBoost Classifier Select From Model'
    component_type = ComponentTypes.FEATURE_SELECTION_CLASSIFIER
    _needs_fitting = True
    hyperparameter_ranges = {
        "percent_features": Real(.01, 1),
        "threshold": ['mean', -np.inf]
    }

    def __init__(self, number_features=None, n_estimators=10, max_depth=None, eta=0.03,
                 percent_features=0.5, threshold=-np.inf, n_jobs=-1, random_state=0):
        pass
        # max_features = None
        # if number_features:
        #     max_features = max(1, int(percent_features * number_features))
        # parameters = {"percent_features": percent_features,
        #               "threshold": threshold}
        # try:
        #     import catboost
        # except ImportError:
        #     raise ImportError("catboost is not installed. Please install using `pip install catboost.`")
        # estimator = catboost.CatBoostClassifier(n_estimators=n_estimators,
        #                                         eta=eta,
        #                                         max_depth=max_depth,
        #                                         silent=True,
        #                                         random_state=random_state)
        # feature_selection = SkSelect(estimator=estimator,
        #                              max_features=max_features,
        #                              threshold=threshold)
        # super().__init__(parameters=parameters,
        #                  component_obj=feature_selection,
        #                  random_state=random_state)
