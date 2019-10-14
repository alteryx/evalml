from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimator import Estimator
from sklearn.linear_model import LinearRegression as SKLinearRegression

class LinearRegressor(Estimator):
    """Linear Regressor"""
    hyperparameters = {}

    def __init__(self, n_jobs=-1):
        self.name = "Linear Regressor"
        self.component_type = ComponentTypes.REGRESSOR
        self._component_obj = SKLinearRegression()
        self.parameters = {}
        super().__init__(name=self.name, component_type=self.component_type, parameters=self.parameters, component_obj=self._component_obj)
