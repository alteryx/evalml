from evalml.pipelines import BinaryClassificationPipeline


class CatBoostBinaryClassificationPipeline(BinaryClassificationPipeline):
    """
    CatBoost Pipeline for binary classification.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Classifier']
    supported_problem_types = ['binary']
    custom_hyperparameters = {
        "impute_strategy": ["most_frequent"],
    }

    def __init__(self, parameters):
        # note: impute_strategy must support both string and numeric data
        super().__init__(parameters=parameters)
