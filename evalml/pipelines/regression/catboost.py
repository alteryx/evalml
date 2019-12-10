from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import CatBoostRegressor, SimpleImputer
from evalml.problem_types import ProblemTypes


class CatBoostRegressionPipeline(PipelineBase):
    """CatBoost Pipeline for regression problems"""
    name = "CatBoost Regressor w/ Simple Imputer"
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.REGRESSION]
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 16),
    }

    def __init__(self, objective, impute_strategy, n_estimators, eta, max_depth, number_features,
                 n_jobs=-1, random_state=0):
        # note: impute_strategy must support both string and numeric data
        imputer = SimpleImputer(impute_strategy=impute_strategy)
        estimator = CatBoostRegressor(n_estimators=n_estimators, eta=eta, max_depth=max_depth, random_state=random_state)
        super().__init__(objective=objective,
                         component_list=[imputer, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
