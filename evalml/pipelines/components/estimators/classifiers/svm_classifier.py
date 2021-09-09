"""Support Vector Machine Classifier."""
import numpy as np
from sklearn.svm import SVC
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class SVMClassifier(Estimator):
    """Support Vector Machine Classifier.

    Args:
        C (float): The regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. Defaults to 1.0.
        kernel ({"poly", "rbf", "sigmoid"}): Specifies the kernel type to be used in the algorithm. Defaults to "rbf".
        gamma ({"scale", "auto"} or float): Kernel coefficient for "rbf", "poly" and "sigmoid". Defaults to "auto".
            - If gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma
            - If "auto" (default), uses 1 / n_features
        probability (boolean): Whether to enable probability estimates. Defaults to True.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "SVM Classifier"
    hyperparameter_ranges = {
        "C": Real(0, 10),
        "kernel": ["poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }
    """{
        "C": Real(0, 10),
        "kernel": ["poly", "rbf", "sigmoid"],
        "gamma": ["scale", "auto"],
    }"""
    model_family = ModelFamily.SVM
    """ModelFamily.SVM"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        gamma="auto",
        probability=True,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "C": C,
            "kernel": kernel,
            "gamma": gamma,
            "probability": probability,
        }
        parameters.update(kwargs)
        svm_classifier = SVC(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters, component_obj=svm_classifier, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance only works with linear kernels.

        If the kernel isn't linear, we return a numpy array of zeros.

        Returns:
            Feature importance of fitted SVM classifier or a numpy array of zeroes if the kernel is not linear.
        """
        if self._parameters["kernel"] != "linear":
            return np.zeros(self._component_obj.n_features_in_)
        else:
            return self._component_obj.coef_
