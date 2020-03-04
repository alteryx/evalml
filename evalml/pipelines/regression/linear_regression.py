from evalml.model_family import ModelFamily
from evalml.pipelines import PipelineBase


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for regression problems"""
    name = "Linear Regressor w/ Simple Imputer + One Hot Encoder + Standard Scaler"
    model_family = ModelFamily.LINEAR_MODEL
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'Standard Scaler', 'Linear Regressor']
    problem_types = ['regression']

    hyperparameters = {
        'impute_strategy': ['most_frequent', 'mean', 'median'],
        'normalize': [False, True],
        'fit_intercept': [False, True]
    }

    def __init__(self, objective, parameters, number_features=0, random_state=0, n_jobs=-1):
        super().__init__(objective=objective,
                         parameters=parameters,
                         number_features=number_features,
                         random_state=random_state,
                         n_jobs=n_jobs)
