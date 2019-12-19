from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (
    LinearRegressor,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)


class LinearRegressionPipeline(PipelineBase):
    """Linear Regression Pipeline for regression problems"""

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
