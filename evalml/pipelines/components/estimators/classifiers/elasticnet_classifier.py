"""Elastic Net Classifier. Uses Logistic Regression with elasticnet penalty as the base estimator."""
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetClassifier(Estimator):
    """Elastic Net Classifier. Uses Logistic Regression with elasticnet penalty as the base estimator.

    Args:
        penalty ({"l1", "l2", "elasticnet", "none"}): The norm used in penalization. Defaults to "elasticnet".
        C (float): Inverse of regularization strength. Must be a positive float. Defaults to 1.0.
        l1_ratio (float): The mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2',
            while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2. Defaults to 0.15.
        multi_class ({"auto", "ovr", "multinomial"}): If the option chosen is "ovr", then a binary problem is fit for each label.
            For "multinomial" the loss minimised is the multinomial loss fit across the entire probability distribution,
            even when the data is binary. "multinomial" is unavailable when solver="liblinear".
            "auto" selects "ovr" if the data is binary, or if solver="liblinear", and otherwise selects "multinomial". Defaults to "auto".
        solver ({"newton-cg", "lbfgs", "liblinear", "sag", "saga"}): Algorithm to use in the optimization problem.
            For small datasets, "liblinear" is a good choice, whereas "sag" and "saga" are faster for large ones.
            For multiclass problems, only "newton-cg", "sag", "saga" and "lbfgs" handle multinomial loss; "liblinear" is limited to one-versus-rest schemes.

            - "newton-cg", "lbfgs", "sag" and "saga" handle L2 or no penalty
            - "liblinear" and "saga" also handle L1 penalty
            - "saga" also supports "elasticnet" penalty
            - "liblinear" does not support setting penalty='none'

            Defaults to "saga".
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Elastic Net Classifier"
    hyperparameter_ranges = {"C": Real(0.01, 10), "l1_ratio": Real(0, 1)}
    """{
        "C": Real(0.01, 10),
        "l1_ratio": Real(0, 1)
    }"""
    model_family = ModelFamily.LINEAR_MODEL
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
        penalty="elasticnet",
        C=1.0,
        l1_ratio=0.15,
        multi_class="auto",
        solver="saga",
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "penalty": penalty,
            "C": C,
            "l1_ratio": l1_ratio,
            "n_jobs": n_jobs,
            "multi_class": multi_class,
            "solver": solver,
        }
        parameters.update(kwargs)
        lr_classifier = LogisticRegression(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters, component_obj=lr_classifier, random_seed=random_seed
        )

    def fit(self, X, y):
        """Fits ElasticNet classifier component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features].
            y (pd.Series): The target training data of length [n_samples].

        Returns:
            self
        """
        warnings.filterwarnings("ignore", message="The max_iter was reached")
        return super().fit(X, y)

    @property
    def feature_importance(self):
        """Feature importance for fitted ElasticNet classifier."""
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return coef_[0]
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)
