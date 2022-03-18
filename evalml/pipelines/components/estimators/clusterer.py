"""A component that fits and predicts given data."""
from abc import abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.model_family import ModelFamily
from evalml.pipelines.components.estimators import Estimator
from evalml.utils import infer_feature_types


class Clusterer(Estimator):
    """A component that fits and predicts given data in unsupervised contexts.

    To implement a new Clusterer, define your own class which is a subclass of Clusterer, with a given name.
    Define an `__init__` method which sets up any necessary state and objects. Make sure your `__init__` only
    uses standard keyword arguments and calls `super().__init__()` with a parameters dict. You may also override the
    `fit`, `transform`, `fit_transform` and other methods in this class if appropriate.

    To see some examples, check out the definitions of any Clusterer component subclass.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    model_family = ModelFamily.NONE
    """ModelFamily.NONE"""

    modifies_features = True
    modifies_target = False
    training_only = False

    @property
    @classmethod
    @abstractmethod
    def model_family(cls):
        """Returns ModelFamily of this component."""

    @property
    @classmethod
    @abstractmethod
    def supported_problem_types(cls):
        """Problem types this estimator supports."""

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        self.input_feature_names = None
        super().__init__(
            parameters=parameters,
            component_obj=component_obj,
            random_seed=random_seed,
            **kwargs,
        )

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (pd.DataFrame): Data of shape [n_samples, n_features]. Must be the original data the clusterer was trained on.

        Returns:
            pd.Series: Predicted values.

        Raises:
            MethodPropertyNotFoundError: If estimator does not have a predict method or a component_obj that contains labels.
        """
        try:
            predictions = self._component_obj.labels_
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Clusterer requires a predict method or a component_obj that contains labels"
            )
        return infer_feature_types(predictions)
