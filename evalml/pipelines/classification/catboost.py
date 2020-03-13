from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class CatBoostClassificationPipeline(PipelineBase):
    """
    CatBoost Pipeline for both binary and multiclass classification.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    Note: impute_strategy must support both string and numeric data
    """
    _name = "CatBoost Classification Pipeline"
    model_type = ModelTypes.CATBOOST
    component_graph = ['Simple Imputer', 'CatBoost Classifier']
    problem_types = ['binary', 'multiclass']
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, parameters, objective):

        # note: impute_strategy must support both string and numeric data
        super().__init__(parameters=parameters,
                         objective=objective)
