from skopt.space import Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase, PipelineTemplate
from evalml.problem_types import ProblemTypes


class LogisticRegressionPipeline(PipelineBase):
    """Logistic Regression Pipeline for both binary and multiclass classification"""
    name = "Logistic Regression Classifier w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    hyperparameters = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
        "impute_strategy": ["mean", "median", "most_frequent"],
    }

    def __init__(self, objective, parameters):
        component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Logistic Regression Classifier']
        supported_problem_types = ['binary', 'multiclass']
        # template = PipelineTemplate(component_graph=component_graph, supported_problem_types=supported_problem_types)
        super().__init__(objective=objective,
                         component_graph=component_graph,
                         supported_problem_types=supported_problem_types,
                         parameters=parameters)
