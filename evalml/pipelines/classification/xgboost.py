from evalml.pipelines import PipelineBase
from evalml.problem_types import ProblemTypes


class XGBoostPipeline(PipelineBase):
    """XGBoost Pipeline for both binary and multiclass classification"""
    _name = "XGBoost Classification Pipeline"
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'RF Classifier Select From Model', 'XGBoost Classifier']
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, parameters, objective, random_state=0):
        super().__init__(parameters=parameters,
                         objective=objective,
                         random_state=random_state)
