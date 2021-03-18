import copy
from abc import ABC, abstractmethod

import cloudpickle

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.component_base_meta import ComponentBaseMeta
from evalml.utils import (
    _convert_woodwork_types_wrapper,
    classproperty,
    get_logger,
    infer_feature_types,
    log_subtitle,
    safe_repr
)

logger = get_logger(__file__)


class ComponentBase(ABC, metaclass=ComponentBaseMeta):
    """Base class for all components."""
    _default_parameters = None

    def __init__(self, parameters=None, component_obj=None, random_seed=0, **kwargs):
        self.random_seed = random_seed
        self._component_obj = component_obj
        self._parameters = parameters or {}
        self._is_fitted = False

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns string name of this component"""

    @property
    @classmethod
    @abstractmethod
    def model_family(cls):
        """Returns ModelFamily of this component"""

    @classproperty
    def needs_fitting(self):
        """Returns boolean determining if component needs fitting before
            calling predict, predict_proba, transform, or feature_importances.
            This can be overridden to False for components that do not need to be fit
            or whose fit methods do nothing."""
        return True

    @property
    def parameters(self):
        """Returns the parameters which were used to initialize the component"""
        return copy.copy(self._parameters)

    @classproperty
    def default_parameters(cls):
        """Returns the default parameters for this component.

         Our convention is that Component.default_parameters == Component().parameters.

         Returns:
             dict: default parameters for this component.
        """

        if cls._default_parameters is None:
            cls._default_parameters = cls().parameters

        return cls._default_parameters

    def clone(self):
        """Constructs a new component with the same parameters and random state.

        Returns:
            A new instance of this component with identical parameters and random state.
        """
        return self.__class__(**self.parameters, random_seed=self.random_seed)

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (list, ww.DataTable, pd.DataFrame or np.ndarray): The input training data of shape [n_samples, n_features]
            y (list, ww.DataColumn, pd.Series, np.ndarray, optional): The target training data of length [n_samples]

        Returns:
            self
        """
        X = infer_feature_types(X)
        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        if y is not None:
            y = infer_feature_types(y)
            y = _convert_woodwork_types_wrapper(y.to_series())
        try:
            self._component_obj.fit(X, y)
            return self
        except AttributeError:
            raise MethodPropertyNotFoundError("Component requires a fit method or a component_obj that implements fit")

    def describe(self, print_name=False, return_dict=False):
        """Describe a component and its parameters

        Arguments:
            print_name(bool, optional): whether to print name of component
            return_dict(bool, optional): whether to return description as dictionary in the format {"name": name, "parameters": parameters}

        Returns:
            None or dict: prints and returns dictionary
        """
        if print_name:
            title = self.name
            log_subtitle(logger, title)
        for parameter in self.parameters:
            parameter_str = ("\t * {} : {}").format(parameter, self.parameters[parameter])
            logger.info(parameter_str)
        if return_dict:
            component_dict = {"name": self.name}
            component_dict.update({"parameters": self.parameters})
            return component_dict

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves component at file path

        Arguments:
            file_path (str): Location to save file
            pickle_protocol (int): The pickle data stream format.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads component at file path

        Arguments:
            file_path (str): Location to load file

        Returns:
            ComponentBase object
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        random_seed_eq = self.random_seed == other.random_seed
        if not random_seed_eq:
            return False
        attributes_to_check = ['_parameters', '_is_fitted']
        for attribute in attributes_to_check:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __str__(self):
        return self.name

    def __repr__(self):
        parameters_repr = ', '.join([f'{key}={safe_repr(value)}' for key, value in self.parameters.items()])
        return f'{(type(self).__name__)}({parameters_repr})'
