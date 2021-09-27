"""Vowpal Wabbit Classifier."""
from skopt.space import Real
from vowpalwabbit.sklearn_vw import VWClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes


class VowpalWabbitBinaryClassifier(Estimator):
    """Vowpal Wabbit Binary Classifier.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Vowpal Wabbit Binary Classifier"
    hyperparameter_ranges = {
        "learning_rate": Real(0.000001, 10),
        # "decay_learning_rate": Real(0.000001, 1.0),
    }
    """"""
    model_family = ModelFamily.VOWPAL_WABBIT
    """ModelFamily.VOWPAL_WABBIT"""
    supported_problem_types = [
        ProblemTypes.BINARY,
        # ProblemTypes.MULTICLASS,  # VWMultiClassifier? Or supported by this?
        ProblemTypes.TIME_SERIES_BINARY,
        # ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_BINARY,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""

    def __init__(
        self,
        learning_rate=0.5,
        # decay_learning_rate=0.95,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "learning_rate": learning_rate,
            # "decay_learning_rate": decay_learning_rate,
        }
        parameters.update(kwargs)
        vw_classifier = VWClassifier(learning_rate=0.5)
        super().__init__(
            parameters=parameters, component_obj=vw_classifier, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance."""
        raise ValueError("not yet impl")
