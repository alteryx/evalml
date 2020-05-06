from evalml.pipelines import RegressionPipeline


class CatBoostRegressionPipeline(RegressionPipeline):
    """
    CatBoost Pipeline for regression problems.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/

    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Regressor']
    custom_hyperparameters = {
        "impute_strategy": ["most_frequent"],
    }
