from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import BinaryClassificationPipeline


class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
    """Logistic Regression Pipeline for binary classification"""
    name = "Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['binary', 'multiclass']

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }
