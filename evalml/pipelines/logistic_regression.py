import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt.space import Real

from .pipeline_base import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    name = "LogisticRegression w/ Imputation"
    model_type = "linear_model"

    hyperparameters = {
        "penalty": ["l2", None],
        "C": Real(.01, 1),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, penalty, C, impute_strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):
        imputer = SimpleImputer(strategy=impute_strategy)

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
