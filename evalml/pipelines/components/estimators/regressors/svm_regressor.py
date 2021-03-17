import numpy as np
from sklearn.svm import SVR
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class SVMRegressor(Estimator):
    """Support Vector Machine Regressor."""
    name = "SVM Regressor"
    hyperparameter_ranges = {
        "C": Real(0, 10),
        "kernel": ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "gamma": ["scale", "auto"]
    }
    model_family = ModelFamily.SVM
    supported_problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]

    def __init__(self,
                 C=1.0,
                 kernel="rbf",
                 gamma="scale",
                 random_seed=0,
                 **kwargs):
        parameters = {"C": C,
                      "kernel": kernel,
                      "gamma": gamma}
        parameters.update(kwargs)

        # SVR doesn't take a random_state arg
        svm_regressor = SVR(**parameters)
        super().__init__(parameters=parameters,
                         component_obj=svm_regressor,
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
