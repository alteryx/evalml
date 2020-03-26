from sklearn.linear_model import LinearRegression as SKLinearRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LinearRegressor(Estimator):
    """Linear Regressor"""
    name = "Linear Regressor"
    hyperparameter_ranges = {
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }
    model_family = ModelFamily.LINEAR_MODEL
    problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, parameters={}, component_obj=None, random_state=0):
        linear_regressor = SKLinearRegression(fit_intercept=parameters.get('fit_intercept', True),
                                              normalize=parameters.get('normalize', False),
                                              n_jobs=parameters.get('n_jobs', -1))
        super().__init__(parameters=parameters,
                         component_obj=linear_regressor,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.coef_
