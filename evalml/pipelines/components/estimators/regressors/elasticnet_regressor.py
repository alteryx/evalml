"""Elastic Net Regressor."""
import pandas as pd
from sklearn.linear_model import ElasticNet as SKElasticNet
from skopt.space import Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class ElasticNetRegressor(Estimator):
    """Elastic Net Regressor.

    Args:
        alpha (float): Constant that multiplies the penalty terms. Defaults to 0.0001.
        l1_ratio (float): The mixing parameter, with 0 <= l1_ratio <= 1. Only used if penalty='elasticnet'. Setting l1_ratio=0 is equivalent to using penalty='l2',
            while setting l1_ratio=1 is equivalent to using penalty='l1'. For 0 < l1_ratio <1, the penalty is a combination of L1 and L2. Defaults to 0.15.
        max_iter (int): The maximum number of iterations. Defaults to 1000.
        normalize (boolean): If True, the regressors will be normalized before regression by subtracting the mean
            and dividing by the l2-norm. Defaults to False.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Elastic Net Regressor"
    hyperparameter_ranges = {
        "alpha": Real(0, 1),
        "l1_ratio": Real(0, 1),
    }
    """{
        "alpha": Real(0, 1),
        "l1_ratio": Real(0, 1),
    }"""
    model_family = ModelFamily.LINEAR_MODEL
    """ModelFamily.LINEAR_MODEL"""
    supported_problem_types = [
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]
    """[
        ProblemTypes.REGRESSION,
        ProblemTypes.TIME_SERIES_REGRESSION,
    ]"""

    def __init__(
        self,
        alpha=0.0001,
        l1_ratio=0.15,
        max_iter=1000,
        normalize=False,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "normalize": normalize,
        }
        parameters.update(kwargs)
        en_regressor = SKElasticNet(random_state=random_seed, **parameters)
        super().__init__(
            parameters=parameters,
            component_obj=en_regressor,
            random_seed=random_seed,
        )

    @property
    def feature_importance(self):
        """Feature importance for fitted ElasticNet regressor."""
        return pd.Series(self._component_obj.coef_)
