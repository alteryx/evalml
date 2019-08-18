import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt.space import Real

from .pipeline_base import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    name = "LogisticRegression w/ imputation + scaling"
    model_type = "linear_model"

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, penalty, C, impute_strategy,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=impute_strategy)

        estimator = LogisticRegression(random_state=random_state,
                                       penalty=penalty,
                                       C=C,
                                       solver="lbfgs",
                                       n_jobs=-1)


        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("scaler", StandardScaler()),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)
