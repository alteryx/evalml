from skopt.space import Integer, Real

from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase


class RFRegressionPipeline(PipelineBase):
    """Random Forest Pipeline for regression problems"""
    _name = "Random Forest Regressor w/ Simple Imputer + One Hot Encoder + RF Regressor Select From Model"
    model_type = ModelTypes.RANDOM_FOREST
    component_graph = ['Simple Imputer', 'One Hot Encoder', 'RF Regressor Select From Model', 'Random Forest Regressor']
    problem_types = ['regression']

    hyperparameters = {
        "n_estimators": Integer(10, 1000),
        "max_depth": Integer(1, 32),
        "impute_strategy": ["mean", "median", "most_frequent"],
        "percent_features": Real(.01, 1)
    }

    def __init__(self, objective, parameters, number_features=0, random_state=0, n_jobs=-1):
        super().__init__(objective=objective,
                         parameters=parameters,
                         number_features=number_features,
                         random_state=random_state,
                         n_jobs=n_jobs)
