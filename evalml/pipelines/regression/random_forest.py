import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skopt.space import Integer, Real

from evalml.pipelines import PipelineBase


class RFRegressionPipeline(PipelineBase):
    name = "Random Forest w/ imputation"
    model_type = "random_forest"
    problem_type = "regression"

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, n_estimators, max_depth, impute_strategy, percent_features,
                 number_features, n_jobs=1, random_state=0):

        imputer = SimpleImputer(strategy=impute_strategy)

        estimator = RandomForestRegressor(random_state=random_state,
                                          n_estimators=n_estimators,
                                          max_depth=max_depth,
                                          n_jobs=n_jobs)

        feature_selection = SelectFromModel(
            estimator=estimator,
            max_features=max(1, int(percent_features * number_features)),
            threshold=-np.inf
        )

        self.pipeline = Pipeline(
            [("imputer", imputer),
             ("feature_selection", feature_selection),
             ("estimator", estimator)]
        )

        super().__init__(objective=objective, random_state=random_state)
