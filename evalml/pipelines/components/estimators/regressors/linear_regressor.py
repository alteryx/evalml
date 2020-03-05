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

    def __init__(self, fit_intercept=True, normalize=False, n_jobs=-1):
        parameters = {
            'fit_intercept': fit_intercept,
            'normalize': normalize
        }
        linear_regressor = SKLinearRegression(fit_intercept=fit_intercept,
                                              normalize=normalize,
                                              n_jobs=n_jobs)
        super().__init__(parameters=parameters,
                         component_obj=linear_regressor,
                         random_state=0)

    @property
    def feature_importances(self):
        return self._component_obj.coef_
