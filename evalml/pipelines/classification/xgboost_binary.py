from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import BinaryClassificationPipeline


class XGBoostBinaryPipeline(BinaryClassificationPipeline):
    """XGBoost Pipeline for binary classification"""
    name = "XGBoost Classifier w/ One Hot Encoder + Simple Imputer + RF Classifier Select From Model"
    model_type = ModelTypes.XGBOOST
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']
    problem_types = ['binary']

    hyperparameters = {
        "eta": Real(0, 1),
        "min_child_weight": Real(1, 10),
        "max_depth": Integer(1, 20),
        "n_estimators": Integer(1, 1000),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1),
    }

    def __init__(self, parameters):
        super().__init__(parameters=parameters)
