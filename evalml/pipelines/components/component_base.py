"""Base class for all components."""
import copy
from abc import ABC, abstractmethod

import cloudpickle

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.component_base_meta import ComponentBaseMeta
from evalml.utils import classproperty, infer_feature_types, log_subtitle, safe_repr
from evalml.utils.logger import get_logger


class ComponentBase(ABC, metaclass=ComponentBaseMeta):
    """Base class for all components.

    Args:
        parameters (dict): Dictionary of parameters for the component. Defaults to None.
        component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
        random_seed (int): Seed for the random number generator. Defaults to 0.
    """

    _default_parameters = None
    _can_be_used_for_fast_partial_dependence = True

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        """Base class for all components.

        Args:
            parameters (dict): Dictionary of parameters for the component. Defaults to None.
            component_obj (obj): Third-party objects useful in component implementation. Defaults to None.
            random_seed (int): Seed for the random number generator. Defaults to 0.
        """
        self.random_seed = random_seed
        self._component_obj = component_obj
        self._parameters = parameters or {}
        self._is_fitted = False

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns string name of this component."""

    @property
    @classmethod
    @abstractmethod
    def modifies_features(cls):
        """Returns whether this component modifies (subsets or transforms) the features variable during transform.

        For Estimator objects, this attribute determines if the return
        value from `predict` or `predict_proba` should be used as
        features or targets.
        """

    @property
    @classmethod
    @abstractmethod
    def modifies_target(cls):
        """Returns whether this component modifies (subsets or transforms) the target variable during transform.

        For Estimator objects, this attribute determines if the return
        value from `predict` or `predict_proba` should be used as
        features or targets.
        """

    @property
    @classmethod
    @abstractmethod
    def training_only(cls):
        """Returns whether or not this component should be evaluated during training-time only, or during both training and prediction time."""

    @classproperty
    def needs_fitting(self):
        """Returns boolean determining if component needs fitting before calling predict, predict_proba, transform, or feature_importances.

        This can be overridden to False for components that do not need to be fit or whose fit methods do nothing.

        Returns:
            True.
        """
        return True

    @property
    def parameters(self):
        """Returns the parameters which were used to initialize the component."""
        return copy.copy(self._parameters)

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for this component.

        Our convention is that Component.default_parameters == Component().parameters.

        Returns:
            dict: Default parameters for this component.
        """
        if cls._default_parameters is None:
            cls._default_parameters = cls().parameters

        return cls._default_parameters

    @classproperty
    def _supported_by_list_API(cls):
        return not cls.modifies_target

    def _handle_partial_dependence_fast_mode(
        self,
        pipeline_parameters,
        X=None,
        target=None,
    ):
        """Determines whether or not a component can be used with partial dependence's fast mode.

        Args:
            pipeline_parameters (dict): Pipeline parameters that will be used to create the pipelines
                used in partial dependence fast mode.
            X (pd.DataFrame, optional): Holdout data being used for partial dependence calculations.
            target (str, optional): The target whose values we are trying to predict.
        """
        if self._can_be_used_for_fast_partial_dependence:
            return pipeline_parameters

        raise TypeError(
            f"Component {self.name} cannot run partial dependence fast mode.",
        )

    def clone(self):
        """Constructs a new component with the same parameters and random state.

        Returns:
            A new instance of this component with identical parameters and random state.
        """
        return self.__class__(**self.parameters, random_seed=self.random_seed)

    def fit(self, X, y=None):
        """Fits component to data.

        Args:
            X (pd.DataFrame): The input training data of shape [n_samples, n_features]
            y (pd.Series, optional): The target training data of length [n_samples]

        Returns:
            self

        Raises:
            MethodPropertyNotFoundError: If component does not have a fit method or a component_obj that implements fit.
        """
        X = infer_feature_types(X)
        if y is not None:
            y = infer_feature_types(y)
        try:
            self._component_obj.fit(X, y)
            return self
        except AttributeError:
            raise MethodPropertyNotFoundError(
                "Component requires a fit method or a component_obj that implements fit",
            )

    def describe(self, print_name=False, return_dict=False):
        """Describe a component and its parameters.

        Args:
            print_name(bool, optional): whether to print name of component
            return_dict(bool, optional): whether to return description as dictionary in the format {"name": name, "parameters": parameters}

        Returns:
            None or dict: Returns dictionary if return_dict is True, else None.
        """
        logger = get_logger(f"{__name__}.describe")
        if print_name:
            title = self.name
            log_subtitle(logger, title)
        for parameter in self.parameters:
            parameter_str = ("\t * {} : {}").format(
                parameter,
                self.parameters[parameter],
            )
            logger.info(parameter_str)
        if return_dict:
            component_dict = {"name": self.name}
            component_dict.update({"parameters": self.parameters})
            return component_dict

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves component at file path.

        Args:
            file_path (str): Location to save file.
            pickle_protocol (int): The pickle data stream format.
        """
        with open(file_path, "wb") as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads component at file path.

        Args:
            file_path (str): Location to load file.

        Returns:
            ComponentBase object
        """
        with open(file_path, "rb") as f:
            return cloudpickle.load(f)

    def __eq__(self, other):
        """Check for equality."""
        if not isinstance(other, self.__class__):
            return False
        random_seed_eq = self.random_seed == other.random_seed
        if not random_seed_eq:
            return False
        attributes_to_check = ["_parameters", "_is_fitted"]
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __str__(self):
        """String representation of a component."""
        return self.name

    def __repr__(self):
        """String representation of a component."""
        parameters_repr = ", ".join(
            [f"{key}={safe_repr(value)}" for key, value in self.parameters.items()],
        )
        return f"{(type(self).__name__)}({parameters_repr})"
