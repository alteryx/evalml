from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""
    _name = "Random Forest Classification Pipeline"
    model_type = ModelTypes.RANDOM_FOREST
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
