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
        "loss_function": ["squared", "classic", "hinge", "logistic"],
        "learning_rate": Real(0.0000001, 10),
        "decay_learning_rate": Real(0.0000001, 1.0),
        "power_t": Real(0.01, 1.0),
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
        loss_function="logistic",
        learning_rate=0.5,
        decay_learning_rate=0.95,
        power_t=1.0,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "decay_learning_rate": decay_learning_rate,
            "power_t": power_t,
        }
        parameters.update(kwargs)
        vw_classifier = VWRegessor(**parameters)
        super().__init__(
            parameters=parameters, component_obj=vw_classifier, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance."""
        raise ValueError("not yet impl")
