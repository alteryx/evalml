from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    name = "Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder + Standard Scaler"
    model_family = ModelFamily.LINEAR_MODEL
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }
