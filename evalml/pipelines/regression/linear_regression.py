from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for regression problems"""
    name = "Linear Regressor w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Linear Regressor']
    problem_types = ['regression']

    hyperparameters = {
        'impute_strategy': ['most_frequent', 'mean', 'median'],
        'normalize': [False, True],
        'fit_intercept': [False, True]
    }

    def __init__(self, objective, parameters):
        super().__init__(objective=objective,
                         parameters=parameters,
                         component_graph=self.__class__.component_graph,
                         problem_types=self.__class__.problem_types)
