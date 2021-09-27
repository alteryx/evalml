"""Vowpal Wabbit Classifier."""
from skopt.space import Real
from vowpalwabbit.sklearn_vw import VWRegessor

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class VowpalWabbitRegressor(Estimator):
    """Vowpal Wabbit Regressor.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Vowpal Wabbit Regressor"
    hyperparameter_ranges = {
        # "learning_rate": Real(0.000001, 1),
    }
    """"""
    model_family = ModelFamily.VOWPAL_WABBIT
    """ModelFamily.VOWPAL_WABBIT"""
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
        learning_rate=0.1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "learning_rate": learning_rate,
        }
        parameters.update(kwargs)
        vw_regressor = VWRegessor(**parameters)
        super().__init__(
            parameters=parameters, component_obj=vw_regressor, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance."""
        raise ValueError("not yet impl")
