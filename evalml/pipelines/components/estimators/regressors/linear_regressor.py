from sklearn.linear_model import LinearRegression as SKLinearRegression

from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class LinearRegressor(Estimator):
    """Linear Regressor"""
    hyperparameters = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }

    def __init__(self, fit_intercept=True, normalize=False, n_jobs=-1):
        self.name = "Linear Regressor"
        self.component_type = ComponentTypes.REGRESSOR
        self._component_obj = SKLinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.parameters = {
            'fit_intercept': self.fit_intercept,
            'normalize': self.normalize
        }
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, component_obj=self._component_obj)
