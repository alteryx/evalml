from skopt.space import Real

from evalml.pipelines import MulticlassClassificationPipeline


class LogisticRegressionMulticlassPipeline(MulticlassClassificationPipeline):
    """Logistic Regression Pipeline for multiclass classification"""
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Logistic Regression Classifier']
    problem_types = ['multiclass']

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }
