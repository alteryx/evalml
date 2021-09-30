"""Vowpal Wabbit Classifiers."""
from skopt.space import Real
from vowpalwabbit.sklearn_vw import VWClassifier, VWMultiClassifier

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
import numpy as np

class VowpalWabbitBaseClassifier(Estimator):
    """Vowpal Wabbit Base Classifier.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

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
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    ]"""
    _vowpal_wabbit_component = None

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
        vw_classifier = self._vowpal_wabbit_component(**parameters)
        super().__init__(
            parameters=parameters, component_obj=vw_classifier, random_seed=random_seed
        )

    @property
    def feature_importance(self):
        """Feature importance for Vowpal Wabbit classifiers. This is not implemented."""
        coef_ = self._component_obj.get_coefs()
        # binary classification case
        if len(coef_) <= 2:
            return coef_[0]
        else:
            # multiclass classification case
            return np.linalg.norm(coef_, axis=0, ord=2)


class VowpalWabbitBinaryClassifier(VowpalWabbitBaseClassifier):
    """Vowpal Wabbit Binary Classifier.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Vowpal Wabbit Binary Classifier"
    supported_problem_types = [
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    ]
    """[
        ProblemTypes.BINARY,
        ProblemTypes.TIME_SERIES_BINARY,
    ]"""
    _vowpal_wabbit_component = VWClassifier


class VowpalWabbitMulticlassClassifier(Estimator):
    """Vowpal Wabbit Multiclass Classifier.

    Args:
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    name = "Vowpal Wabbit Multiclass Classifier"
    supported_problem_types = [
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]
    """[
        ProblemTypes.MULTICLASS,
        ProblemTypes.TIME_SERIES_MULTICLASS,
    ]"""
    _vowpal_wabbit_component = VWMultiClassifier
