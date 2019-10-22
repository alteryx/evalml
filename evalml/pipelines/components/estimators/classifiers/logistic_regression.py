from sklearn.linear_model import LogisticRegression as LogisticRegression
from skopt.space import Real
import numpy as np
from evalml.model_types import ModelTypes
from evalml.pipelines.components import ComponentTypes
from evalml.pipelines.components.estimators import Estimator


class LogisticRegressionClassifier(Estimator):
    """
    Logistic Regression Classifier
    """

    hyperparameter_ranges = {
        "penalty": ["l2"],
        "C": Real(.01, 10),
    }
    model_type = ModelTypes.LINEAR_MODEL

    def __init__(self, penalty="l2", C=1.0, n_jobs=-1, random_state=0):
        name = "Logistic Regression Classifier"
        component_type = ComponentTypes.CLASSIFIER
        parameters = {"penalty": penalty,
                      "C": C}
        lr_classifier = LogisticRegression(penalty=penalty,
                                           C=C,
                                           random_state=random_state,
                                           multi_class="auto",
                                           solver="lbfgs",
                                           n_jobs=n_jobs)
        n_jobs = n_jobs
        random_state = random_state
        super().__init__(name=name,
                         component_type=component_type,
                         parameters=parameters,
                         component_obj=lr_classifier,
                         needs_fitting=True,
                         random_state=0)

    @property
    def feature_importances(self):
        """Return feature importances. Feature dropped by feaure selection are excluded"""
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_[0]
        else:
            # mutliclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
