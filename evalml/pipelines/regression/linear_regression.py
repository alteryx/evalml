from evalml.model_types import ModelTypes
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    LinearRegressor,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.problem_types import ProblemTypes


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for regression"""
    name = "Linear Regressor w/ One Hot Encoder + Simple Imputer + Standard Scaler"
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.REGRESSION]

    hyperparameters = {
        'impute_strategy': ['most_frequent', 'mean', 'median'],
        'normalize': [False, True],
        'fit_intercept': [False, True]
    }

    def __init__(self, objective, random_state, number_features, impute_strategy, normalize=False, fit_intercept=True, n_jobs=-1):

        imputer = SimpleImputer(impute_strategy=impute_strategy)
        enc = OneHotEncoder()
        scaler = StandardScaler()
        estimator = LinearRegressor(normalize=normalize,
                                    fit_intercept=fit_intercept,
                                    n_jobs=-1)

        super().__init__(objective=objective,
                         component_list=[enc, imputer, scaler, estimator],
                         n_jobs=n_jobs,
                         random_state=random_state)
