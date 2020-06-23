import numpy as np
from sklearn.linear_model import SGDClassifier as SKElasticNetClassifier
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetClassifier(Estimator):
    """Elastic Net Classifier."""
    name = "Elastic Net Classifier"
    hyperparameter_ranges = {
        "alpha": Real(0, 1),
        "l1_ratio": Real(0, 1),
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]

    def __init__(self, alpha=0.5, l1_ratio=0.5, n_jobs=-1, max_iter=1000, random_state=0, **kwargs):
        parameters = {'alpha': alpha,
                      'l1_ratio': l1_ratio,
                      'n_jobs': n_jobs,
                      'max_iter': max_iter}
        parameters.update(kwargs)

        en_classifier = SKElasticNetClassifier(loss="log",
                                               penalty="elasticnet",
                                               random_state=random_state,
                                               **parameters)
        super().__init__(parameters=parameters,
                         component_obj=en_classifier,
                         random_state=random_state)

    @property
    def feature_importances(self):
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_.flatten()
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
