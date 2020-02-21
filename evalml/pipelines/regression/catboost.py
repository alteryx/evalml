from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import RegressionPipeline
from evalml.pipelines.components import CatBoostRegressor, SimpleImputer
from evalml.problem_types import ProblemTypes


class CatBoostRegressionPipeline(RegressionPipeline):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    """
    name = "CatBoost Regressor w/ Simple Imputer"
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.REGRESSION]
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, impute_strategy, n_estimators, eta,
                 max_depth, number_features, bootstrap_type=None,
                 n_jobs=-1, random_state=0):
        # note: impute_strategy must support both string and numeric data
        imputer = SimpleImputer(impute_strategy=impute_strategy)
        estimator = CatBoostRegressor(n_estimators=n_estimators,
                                      eta=eta,
                                      max_depth=max_depth,
                                      bootstrap_type=bootstrap_type,
                                      random_state=random_state)
        super().__init__(component_list=[imputer, estimator],
                         n_jobs=1,
                         random_state=random_state)
