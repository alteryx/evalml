from skopt.space import Integer, Real

from evalml.pipelines import PipelineBase


class CatBoostClassificationPipeline(PipelineBase):
    """
    CatBoost Pipeline for both binary and multiclass classification.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    Note: impute_strategy must support both string and numeric data
    """
    name = "CatBoost Classifier w/ Simple Imputer"
    component_graph = ['Simple Imputer', 'CatBoost Classifier']
    problem_types = ['binary', 'multiclass']
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, objective, parameters, number_features=0, random_state=0, n_jobs=-1):

        # note: impute_strategy must support both string and numeric data
        super().__init__(objective=objective,
                         parameters=parameters,
                         number_features=number_features,
                         random_state=random_state,
                         n_jobs=n_jobs)
