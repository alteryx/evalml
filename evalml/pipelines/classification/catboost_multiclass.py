from evalml.pipelines import MulticlassClassificationPipeline


class CatBoostMulticlassClassificationPipeline(MulticlassClassificationPipeline):
    """
    CatBoost Pipeline for multiclass classification.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    Note: impute_strategy must support both string and numeric data
    """
    component_graph = ['Simple Imputer', 'CatBoost Classifier']
    supported_problem_types = ['multiclass']
    custom_hyperparameters = {
        "impute_strategy": ["most_frequent"],
    }

    def __init__(self, parameters, random_state=0):
        # note: impute_strategy must support both string and numeric data
        super().__init__(parameters=parameters,
                         random_state=random_state)
