from evalml.pipelines import PipelineBase


class CatBoostRegressionPipeline(PipelineBase):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/

    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Regressor']
    problem_types = ['regression']
    _hyperparameters = {
        "impute_strategy": ["most_frequent"],
    }

    def __init__(self, parameters, objective):
        super().__init__(parameters=parameters,
                         objective=objective)
