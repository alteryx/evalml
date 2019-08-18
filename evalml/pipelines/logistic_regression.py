import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real

from .pipeline_base import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    name = "LogisticRegression w/ Imputation"
    model_type = "linear_model"

    hyperparameters = {
        "penalty": ["l1", "l2", "elasticnet"],
        "C": Real(.01, 1),
        "strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, penalty, C, strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=strategy)

        estimator = LogisticRegression(random_state=random_state,
                          penalty=penalty,
                          C=C,
                          solver="lbfgs",
                          n_jobs=-1)

        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=min(1, int(percent_features * number_features)),
            threshold=-np.inf
        )

        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("scaler", StandardScaler()),
             ("feature_selection", feature_selection),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)
