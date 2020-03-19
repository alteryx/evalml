from skopt.space import Integer, Real

from evalml.pipelines import RegressionPipeline


class CatBoostRegressionPipeline(RegressionPipeline):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/

    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Regressor']
    problem_types = ['regression']
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, parameters):
        super().__init__(parameters=parameters)
