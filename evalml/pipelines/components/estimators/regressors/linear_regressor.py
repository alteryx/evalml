from sklearn.linear_model import LinearRegression as SKLinearRegression

from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LinearRegressor(Estimator):
    """Linear Regressor"""
    name = "Linear Regressor"
    component_type = ComponentTypes.REGRESSOR
    hyperparameter_ranges = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }
    model_type = ModelTypes.LINEAR_MODEL
    problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, fit_intercept=True, normalize=False, n_jobs=-1):
        parameters = {
            'fit_intercept': fit_intercept,
            'normalize': normalize
        }
        linear_regressor = SKLinearRegression(fit_intercept=fit_intercept,
                                              normalize=normalize,
                                              n_jobs=n_jobs)
        super().__init__(name=self.name,
                         component_type=self.component_type,
                         parameters=parameters,
                         component_obj=linear_regressor,
                         needs_fitting=True,
                         random_state=0)

    @property
    def feature_importances(self):
        return self._component_obj.coef_
