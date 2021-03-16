import warnings

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
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self, alpha=0.5, l1_ratio=0.5, n_jobs=-1, max_iter=1000,
                 random_seed=0, penalty='elasticnet',
                 **kwargs):
        parameters = {'alpha': alpha,
                      'l1_ratio': l1_ratio,
                      'n_jobs': n_jobs,
                      'max_iter': max_iter,
                      'penalty': penalty}
        if kwargs.get('loss', 'log') != 'log':
            warnings.warn("Parameter loss is being set to 'log' so that ElasticNetClassifier can predict probabilities"
                          f". Originally received '{kwargs['loss']}'.")
        kwargs["loss"] = "log"
        parameters.update(kwargs)
        en_classifier = SKElasticNetClassifier(random_state=random_seed,
                                               **parameters)
        super().__init__(parameters=parameters,
                         component_obj=en_classifier,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_.flatten()
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
