import numpy as np
from sklearn.linear_model import LogisticRegression as LogisticRegression
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LogisticRegressionClassifier(Estimator):
    """
    Logistic Regression Classifier.
    """
    name = "Logistic Regression Classifier"
    hyperparameter_ranges = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
    }
    model_family = ModelFamily.LINEAR_MODEL
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self, penalty="l2", C=1.0, n_jobs=-1, multi_class="auto", solver="lbfgs", random_seed=0, **kwargs):
        parameters = {"penalty": penalty,
                      "C": C,
                      "n_jobs": n_jobs,
                      "multi_class": multi_class,
                      "solver": solver}
        parameters.update(kwargs)
        lr_classifier = LogisticRegression(random_state=random_seed,
                                           **parameters)
        super().__init__(parameters=parameters,
                         component_obj=lr_classifier,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_[0]
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
