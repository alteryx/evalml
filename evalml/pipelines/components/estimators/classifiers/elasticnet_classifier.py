from sklearn.linear_model import SGDClassifier as SKElasticNetClassifier
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes

class ElasticNetClassifier(Estimator):
    """Elastic Net Classifier"""
    name = "Elastic Net Classifier"
    hyperparameter_ranges = {
        "alpha": Real(0,1),
        "l1_ratio": Real(0, 1),
    }
    model_family = ModelFamily.ELASTIC_NET
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, alpha=1.0, l1_ratio=1.0, n_jobs=-1, random_state=0, max_iter=1000):
        parameters = {'alpha': alpha, 
                      'l1_ratio': l1_ratio}
        en_classifier = SKElasticNetClassifier(loss="log",
                                              penalty="elasticnet",
                                              alpha=alpha,
                                              l1_ratio=l1_ratio,
                                              n_jobs=n_jobs,
                                              random_state=random_state,
                                              max_iter=max_iter
                                              )
        super().__init__(parameters=parameters,
                         component_obj=en_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        return self._component_obj.coef_
