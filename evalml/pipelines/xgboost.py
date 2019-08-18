import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real
from xgboost import XGBClassifier

from .pipeline_base import PipelineBase


class XGBoostPipeline(PipelineBase):
    name = "XGBoost w/ Imputation"
    model_type = "xgboost"

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 10),
        "max_depth": Integer(1, 20),
        "strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, eta, min_child_weight, max_depth, strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=strategy)

        estimator = XGBClassifier(
            random_state=random_state,
            eta=eta,
            max_depth=max_depth,
            min_child_weight=min_child_weight
        )

        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=min(1, int(percent_features * number_features)),
            threshold=-np.inf
        )

        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("feature_selection", feature_selection),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)
