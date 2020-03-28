from skopt.space import Real

from evalml.pipelines import BinaryClassificationPipeline


class LogisticRegressionBinaryPipeline(BinaryClassificationPipeline):
    """Logistic Regression Pipeline for binary classification"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    supported_problem_types = ['binary']
    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }
