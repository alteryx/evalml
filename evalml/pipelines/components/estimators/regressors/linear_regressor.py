from sklearn.linear_model import LinearRegression as SKLinearRegression

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class LinearRegressor(Estimator):
    """Linear Regressor"""
    hyperparameter_ranges = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }
    model_type = ModelTypes.LINEAR_MODEL

    def __init__(self, fit_intercept=True, normalize=False, n_jobs=-1):
        name = "Linear Regressor"
        component_type = ComponentTypes.REGRESSOR
        parameters = {
            'fit_intercept': fit_intercept,
            'normalize': normalize
        }
        linear_regressor = SKLinearRegression(fit_intercept=fit_intercept,
                                              normalize=normalize,
                                              n_jobs=n_jobs)
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=linear_regressor,
                         needs_fitting=True,
                         random_state=0)
