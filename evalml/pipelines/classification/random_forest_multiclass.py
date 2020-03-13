from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import MulticlassClassificationPipeline


class RFMulticlassClassificationPipeline(MulticlassClassificationPipeline):
    """Random Forest Pipeline for multiclass classification"""
    name = "Random Forest Classifier w/ One Hot Encoder + Simple Imputer + RF Classifier Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
