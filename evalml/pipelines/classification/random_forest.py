from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class RFClassificationPipeline(PipelineBase):
    """Random Forest Pipeline for both binary and multiclass classification"""
    name = "Random Forest Classifier w/ One Hot Encoder + Simple Imputer + RF Classifier Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'RF Classifier Select From Model', 'Random Forest Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, parameters):
        super().__init__(objective=objective,
                         parameters=parameters,
                         component_graph=self.__class__.component_graph,
                         problem_types=self.__class__.problem_types)
