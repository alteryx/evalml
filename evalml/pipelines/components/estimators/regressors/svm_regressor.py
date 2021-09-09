"""Support Vector Machine Regressor."""
import numpy as np
from sklearn.svm import SVR
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class SVMRegressor(Estimator):
    """Support Vector Machine Regressor.

    Args:
        C (float): The regularization parameter. The strength of the regularization is inversely proportional to C.
            Must be strictly positive. The penalty is a squared l2 penalty. Defaults to 1.0.
        kernel ({"poly", "rbf", "sigmoid"}): Specifies the kernel type to be used in the algorithm. Defaults to "rbf".
        gamma ({"scale", "auto"} or float): Kernel coefficient for "rbf", "poly" and "sigmoid". Defaults to "auto".
            - If gamma='scale' is passed then it uses 1 / (n_features * X.var()) as value of gamma
            - If "auto" (default), uses 1 / n_features
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "SVM Regressor"
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
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(self, C=1.0, kernel="rbf", gamma="auto", random_seed=0, **kwargs):
        parameters = {"C": C, "kernel": kernel, "gamma": gamma}
        parameters.update(kwargs)

        # SVR doesn't take a random_state arg
        svm_regressor = SVR(**parameters)
        super().__init__(
            parameters=parameters, component_obj=svm_regressor, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance of fitted SVM regresor.

        Only works with linear kernels. If the kernel isn't linear, we return a numpy array of zeros.

        Returns:
            The feature importance of the fitted SVM regressor, or an array of zeroes if the kernel is not linear.
        """
        if self._parameters["kernel"] != "linear":
            return np.zeros(self._component_obj.n_features_in_)
        else:
            return self._component_obj.coef_
