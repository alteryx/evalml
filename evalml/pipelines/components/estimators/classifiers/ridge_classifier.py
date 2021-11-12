"""Ridge Classifier."""
import numpy as np
from sklearn.utils.extmath import softmax
from sklearn.linear_model import RidgeClassifier as SKRidgeRegression

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.woodwork_utils import infer_feature_types


class RidgeClassifier(Estimator):
    """Ridge Classifier.

    Args:
        solver ({"newton-cg", "lbfgs", "liblinear", "sag", "saga"}): Algorithm to use in the optimization problem.
            For small datasets, "liblinear" is a good choice, whereas "sag" and "saga" are faster for large ones.
            For multiclass problems, only "newton-cg", "sag", "saga" and "lbfgs" handle multinomial loss; "liblinear" is limited to one-versus-rest schemes.

            - "newton-cg", "lbfgs", "sag" and "saga" handle L2 or no penalty
            - "liblinear" and "saga" also handle L1 penalty
            - "saga" also supports "elasticnet" penalty
            - "liblinear" does not support setting penalty='none'

            Defaults to "lbfgs".
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Ridge Classifier"
    hyperparameter_ranges = {}
    model_family = ModelFamily.ENSEMBLE
    """ModelFamily.LINEAR_MODEL"""
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
        solver="auto",
        n_jobs=-1,
        random_seed=0,
        normalize=True,
        **kwargs,
    ):
        parameters = {
            "solver": solver,
            "normalize": normalize
        }
        parameters.update(kwargs)
        ridge_classifier = SKRidgeRegression(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters, component_obj=ridge_classifier, random_seed=random_seed
        )

    def predict_proba(self, X):
        d = self._component_obj.decision_function(X)
        print(d, type(d))
        if len(d.shape) <= 1:
            d_2d = np.c_[-d, d]
        else:
            d_2d = d
        pred_proba = softmax(d_2d)
        pred_proba = infer_feature_types(pred_proba)
        pred_proba.index = X.index
        return pred_proba

    @property
    def feature_importance(self):
        """Feature importance for fitted logistic regression classifier."""
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_[0]
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
