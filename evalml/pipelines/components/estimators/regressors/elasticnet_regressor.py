from sklearn.linear_model import ElasticNet as SKElasticNet
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetRegressor(Estimator):
    """Elastic Net Regressor."""
    name = "Elastic Net Regressor"
    hyperparameter_ranges = {
        "alpha": Real(0, 1),
        "l1_ratio": Real(0, 1),
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self, alpha=0.5, l1_ratio=0.5, max_iter=1000, normalize=False, random_seed=0, **kwargs):
        parameters = {'alpha': alpha,
                      'l1_ratio': l1_ratio,
                      'max_iter': max_iter,
                      'normalize': normalize}
        parameters.update(kwargs)
        en_regressor = SKElasticNet(random_state=random_seed,
                                    **parameters)
        super().__init__(parameters=parameters,
                         component_obj=en_regressor,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        return self._component_obj.coef_
