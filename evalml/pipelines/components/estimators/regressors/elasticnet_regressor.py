from sklearn.linear_model import ElasticNet as SKElasticNet
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetRegressor(Estimator):
    """Elastic Net Regressor"""
    name = "Elastic Net Regressor"
    hyperparameter_ranges = {
        "alpha": Real(0, 1),
        "l1_ratio": Real(0, 1),
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.REGRESSION]

    def __init__(self, alpha=0.5, l1_ratio=0.5, random_state=0, normalize=False,
                 max_iter=1000, n_jobs=-1):
        parameters = {'alpha': alpha,
                      'l1_ratio': l1_ratio}
        en_regressor = SKElasticNet(alpha=alpha,
                                    l1_ratio=l1_ratio,
                                    random_state=random_state,
                                    normalize=normalize,
                                    max_iter=max_iter
                                    )
        super().__init__(parameters=parameters,
                         component_obj=en_regressor,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.coef_
