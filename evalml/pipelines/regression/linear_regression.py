from evalml.model_types import ModelTypes
from evalml.pipelines import RegressionPipeline


class LinearRegressionPipeline(RegressionPipeline):
    """Linear Regression Pipeline for regression problems"""
    name = "Linear Regressor w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    component_graph = ['One Hot Encoder', 'Simple Imputer', 'Standard Scaler', 'Linear Regressor']
    problem_types = ['regression']

    hyperparameters = {
        'impute_strategy': ['most_frequent', 'mean', 'median'],
        'normalize': [False, True],
        'fit_intercept': [False, True]
    }
