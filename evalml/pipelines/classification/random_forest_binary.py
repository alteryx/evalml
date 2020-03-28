from skopt.space import Integer, Real

from evalml.pipelines import BinaryClassificationPipeline


class RFBinaryClassificationPipeline(BinaryClassificationPipeline):
    """Random Forest Pipeline for binary classification"""
    _name = "Random Forest Binary Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']
    supported_problem_types = ['binary']
    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
