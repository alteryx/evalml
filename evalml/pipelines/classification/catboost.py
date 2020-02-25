from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase, PipelineTemplate
from evalml.problem_types import ProblemTypes


class CatBoostClassificationPipeline(PipelineBase):
    """
    CatBoost Pipeline for both binary and multiclass classification.
    CatBoost is an open-source library and natively supports categorical features.

    For more information, check out https://catboost.ai/
    """
    name = "CatBoost Classifier w/ Simple Imputer"
    model_type = ModelTypes.CATBOOST
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]
    hyperparameters = {
        "impute_strategy": ["most_frequent"],
        "n_estimators": Integer(10, 1000),
        "eta": Real(0, 1),
        "max_depth": Integer(1, 8),
    }

    def __init__(self, objective, parameters):

        # note: impute_strategy must support both string and numeric data
        component_graph = ['Simple Imputer', 'CatBoost Classifier']
        supported_problem_types = ['binary', 'multiclass']
        template = PipelineTemplate(component_graph=component_graph, supported_problem_types=supported_problem_types)
        super().__init__(objective=objective,
                         template=template,
                         parameters=parameters)
