import copy
from abc import ABC, ABCMeta, abstractmethod
from functools import wraps

from evalml.exceptions import (
    ComponentNotYetFittedError,
    MethodPropertyNotFoundError
)
from evalml.utils import (
    classproperty,
    get_logger,
    get_random_state,
    log_subtitle
)

logger = get_logger(__file__)


class ComponentBaseMeta(ABCMeta):
    """Metaclass that overrides creating a new component by wrapping method with validators and setters"""
    from evalml.exceptions import ComponentNotYetFittedError

    NO_FITTING_REQUIRED = ['DropColumns', 'SelectColumns']

    @classmethod
    def set_fit(cls, method):
        @wraps(method)
        def _set_fit(self, X, y=None):
            return_value = method(self, X, y)
            self._is_fitted = True
            return return_value
        return _set_fit

    @classmethod
    def check_for_fit(cls, method):
        """`check_for_fit` wraps a method that validates if `self._is_fitted` is `True`.
            It raises an exception if `False` and calls and returns the wrapped method if `True`.
        """
        @wraps(method)
        def _check_for_fit(self, X=None, y=None):
            klass = type(self).__name__
            if not self._is_fitted and klass not in cls.NO_FITTING_REQUIRED:
                raise ComponentNotYetFittedError(f'This {klass} is not fitted yet. You must fit {klass} before calling {method.__name__}.')
            elif X is None and y is None:
                return method(self)
            elif y is None:
                return method(self, X)
            else:
                return method(self, X, y)
        return _check_for_fit

    def __new__(cls, name, bases, dct):
        if 'predict' in dct:
            dct['predict'] = cls.check_for_fit(dct['predict'])
        if 'predict_proba' in dct:
            dct['predict_proba'] = cls.check_for_fit(dct['predict_proba'])
        if 'transform' in dct:
            dct['transform'] = cls.check_for_fit(dct['transform'])
        if 'feature_importance' in dct:
            fi = dct['feature_importance']
            new_fi = property(cls.check_for_fit(fi.__get__), fi.__set__, fi.__delattr__)
            dct['feature_importance'] = new_fi
        if 'fit' in dct:
            dct['fit'] = cls.set_fit(dct['fit'])
        if 'fit_transform' in dct:
            dct['fit_transform'] = cls.set_fit(dct['fit_transform'])
        return super().__new__(cls, name, bases, dct)


class ComponentBase(ABC, metaclass=ComponentBaseMeta):
    """Base class for all components."""
    _default_parameters = None

    def __init__(self, parameters=None, component_obj=None, random_state=0, **kwargs):
        self.random_state = get_random_state(random_state)
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

    def clone(self, random_state=0):
        """Constructs a new component with the same parameters

        Arguments:
            random_state (int): the value to seed the random state with. Can also be a RandomState instance. Defaults to 0.

        Returns:
            A new instance of this component with identical parameters
        """
        return self.__class__(**self.parameters, random_state=random_state)

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
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
