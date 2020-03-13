from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    _name = "Logistic Regression Pipeline"
    model_type = ModelTypes.LINEAR_MODEL
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }
