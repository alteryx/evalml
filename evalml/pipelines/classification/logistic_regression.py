from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    name = "Logistic Regression Classifier w/ Simple Imputer + One Hot Encoder + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, parameters, number_features=0, random_state=0, n_jobs=-1):
        super().__init__(objective=objective,
                         parameters=parameters,
                         number_features=number_features,
                         random_state=random_state,
                         n_jobs=n_jobs)
