from skopt.space import Integer, Real

from evalml.pipelines import MulticlassClassificationPipeline


class RFMulticlassClassificationPipeline(MulticlassClassificationPipeline):
    """Random Forest Pipeline for multiclass classification"""
    _name = "Random Forest Multi-class Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'Random Forest Classifier']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }
