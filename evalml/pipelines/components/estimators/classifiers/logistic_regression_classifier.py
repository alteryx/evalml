"""Logistic Regression Classifier."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class LogisticRegressionClassifier(Estimator):
    """Logistic Regression Classifier.

    Args:
        penalty ({"l1", "l2", "elasticnet", "none"}): The norm used in penalization. Defaults to "l2".
        C (float): Inverse of regularization strength. Must be a positive float. Defaults to 1.0.
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

            Defaults to "lbfgs".
        n_jobs (int): Number of parallel threads used to run xgboost. Note that creating thread contention will significantly slow down the algorithm. Defaults to -1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Logistic Regression Classifier"
    hyperparameter_ranges = {
        "penalty": ["l2"],
        "C": Real(0.01, 10),
    }
    """{
        "penalty": ["l2"],
        "C": Real(0.01, 10),
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
        penalty="l2",
        C=1.0,
        multi_class="auto",
        solver="lbfgs",
        n_jobs=-1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "penalty": penalty,
            "C": C,
            "n_jobs": n_jobs,
            "multi_class": multi_class,
            "solver": solver,
        }
        parameters.update(kwargs)
        lr_classifier = SKLogisticRegression(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=lr_classifier,
            random_seed=random_seed,
        )

    @property
    def feature_importance(self):
        """Feature importance for fitted logistic regression classifier."""
        coef_ = self._component_obj.coef_
        # binary classification case
        if len(coef_) <= 2:
            return pd.Series(coef_[0])
        else:
            # multiclass classification case
            return pd.Series(np.linalg.norm(coef_, axis=0, ord=2))
