"""Vowpal Wabbit Classifiers."""
from abc import abstractmethod

from skopt.space import Integer, Real

from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes
from evalml.utils.gen_utils import import_or_raise


class VowpalWabbitBaseClassifier(Estimator):
    """Vowpal Wabbit Base Classifier.

    Args:
        loss_function (str): Specifies the loss function to use. One of {"squared", "classic", "hinge", "logistic", "quantile"}. Defaults to "logistic".
        learning_rate (float): Boosting learning rate. Defaults to 0.5.
        decay_learning_rate (float): Decay factor for learning_rate. Defaults to 1.0.
        power_t (float): Power on learning rate decay. Defaults to 0.5.
        passes (int): Number of training passes. Defaults to 1.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    hyperparameter_ranges = {
        "loss_function": ["squared", "classic", "hinge", "logistic"],
        "learning_rate": Real(0.0000001, 10),
        "decay_learning_rate": Real(0.0000001, 1.0),
        "power_t": Real(0.01, 1.0),
        "passes": Integer(1, 10),
    }
    """"""
    model_family = ModelFamily.VOWPAL_WABBIT
    """ModelFamily.VOWPAL_WABBIT"""
    _vowpal_wabbit_component = None

    def __init__(
        self,
        loss_function="logistic",
        learning_rate=0.5,
        decay_learning_rate=1.0,
        power_t=0.5,
        passes=1,
        random_seed=0,
        **kwargs,
    ):
        parameters = {
            "loss_function": loss_function,
            "learning_rate": learning_rate,
            "decay_learning_rate": decay_learning_rate,
            "power_t": power_t,
            "passes": passes,
        }
        parameters.update(kwargs)
        vw_class = self._get_component_obj_class()
        vw_classifier = vw_class(**parameters)
        super().__init__(
            parameters=parameters,
            component_obj=vw_classifier,
            random_seed=random_seed,
        )

    @abstractmethod
    def _get_component_obj_class(self):
        """Get the appropriate Vowpal Wabbit class."""

    @property
    def feature_importance(self):
        """Feature importance for Vowpal Wabbit classifiers. This is not implemented."""
        raise NotImplementedError(
            "Feature importance is not implemented for the Vowpal Wabbit classifiers.",
        )


class VowpalWabbitBinaryClassifier(VowpalWabbitBaseClassifier):
    """Vowpal Wabbit Binary Classifier.

    Args:
        loss_function (str): Specifies the loss function to use. One of {"squared", "classic", "hinge", "logistic", "quantile"}. Defaults to "logistic".
        learning_rate (float): Boosting learning rate. Defaults to 0.5.
        decay_learning_rate (float): Decay factor for learning_rate. Defaults to 1.0.
        power_t (float): Power on learning rate decay. Defaults to 0.5.
        passes (int): Number of training passes. Defaults to 1.
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

    def _get_component_obj_class(self):
        vw_error_msg = "Vowpal Wabbit is not installed. Please install using `pip install vowpalwabbit.`"
        vw = import_or_raise("vowpalwabbit", error_msg=vw_error_msg)
        vw_classifier = vw.sklearn_vw.VWClassifier
        return vw_classifier


class VowpalWabbitMulticlassClassifier(VowpalWabbitBaseClassifier):
    """Vowpal Wabbit Multiclass Classifier.

    Args:
        loss_function (str): Specifies the loss function to use. One of {"squared", "classic", "hinge", "logistic", "quantile"}. Defaults to "logistic".
        learning_rate (float): Boosting learning rate. Defaults to 0.5.
        decay_learning_rate (float): Decay factor for learning_rate. Defaults to 1.0.
        power_t (float): Power on learning rate decay. Defaults to 0.5.
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

    def _get_component_obj_class(self):
        vw_error_msg = "Vowpal Wabbit is not installed. Please install using `pip install vowpalwabbit.`"
        vw = import_or_raise("vowpalwabbit.sklearn_vw", error_msg=vw_error_msg)
        vw_classifier = vw.VWMultiClassifier
        return vw_classifier
