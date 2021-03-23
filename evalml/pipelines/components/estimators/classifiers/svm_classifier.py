import numpy as np
from sklearn.svm import SVC
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class SVMClassifier(Estimator):
    """Support Vector Machine Classifier."""
    name = "SVM Classifier"
    hyperparameter_ranges = {
        "C": Real(0, 10),
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "gamma": ["scale", "auto"]
    }
    model_family = ModelFamily.SVM
    supported_problem_types = [ProblemTypes.BINARY, ProblemTypes.MULTICLASS,
                               ProblemTypes.TIME_SERIES_BINARY, ProblemTypes.TIME_SERIES_MULTICLASS]

    def __init__(self,
                 C=1.0,
                 kernel="rbf",
                 gamma="scale",
                 probability=True,
                 random_seed=0,
                 **kwargs):
        parameters = {"C": C,
                      "kernel": kernel,
                      "gamma": gamma,
                      "probability": probability}
        parameters.update(kwargs)
        svm_classifier = SVC(random_state=random_seed,
                             **parameters)
        super().__init__(parameters=parameters,
                         component_obj=svm_classifier,
                         random_seed=random_seed)

    @property
    def feature_importance(self):
        """Feature importance only works with linear kernels.
        If the kernel isn't linear, we return a numpy array of zeros
        """
        if self._parameters['kernel'] != 'linear':
            return np.zeros(self._component_obj.n_features_in_)
        else:
            return self._component_obj.coef_
