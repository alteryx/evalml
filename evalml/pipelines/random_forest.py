import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real

from .pipeline_base import PipelineBase


class RFPipeline(PipelineBase):
    name = "Random Forest w/ Imputation"
    model_type = "random_forest"

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 1000),
        "strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, n_estimators, max_depth, strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=strategy)

        estimator = RandomForestClassifier(random_state=random_state,
                                           n_estimators=n_estimators,
                                           max_depth=max_depth,
                                           n_jobs=n_jobs)

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
