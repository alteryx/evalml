from evalml.pipelines import RegressionPipeline


class CatBoostRegressionPipeline(RegressionPipeline):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/

    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Regressor']
    supported_problem_types = ['regression']
    custom_hyperparameters = {
        "impute_strategy": ["most_frequent"],
    }

    def __init__(self, parameters, random_state=0):
        super().__init__(parameters=parameters,
                         random_state=random_state)
