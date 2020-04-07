from abc import ABC, abstractmethod
from inspect import Parameter, signature, Signature

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.utils import Logger, classproperty, get_random_state
from evalml.pipelines.components.validation_error import ValidationError

logger = Logger()


class ComponentBase(ABC):
    """The abstract base class for all evalml components.

    Please see Transformer and Estimator for examples of how to use this class.
    """

    def __init__(self, component_obj=None, random_state=0):
        if not hasattr(self, 'random_state'):
            self.random_state = get_random_state(random_state)
        self._component_obj = component_obj
        self._parameters = self._introspect_parameters()

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns string name of this component"""
        return NotImplementedError("This component must have `name` as a class variable.")

    @property
    @classmethod
    @abstractmethod
    def model_family(cls):
        """Returns ModelFamily of this component"""
        return NotImplementedError("This component must have `model_family` as a class variable.")

    @property
    def parameters(self):
        return self._parameters

    @classproperty
    def default_parameters(cls):
        return cls._introspect_default_parameters()

    _REQUIRED_SUBCLASS_INIT_ARGS = ['random_state']
    _INVALID_SUBCLASS_INIT_ARGS = ['component_obj']

    @classmethod
    def _introspect_default_parameter(cls, param_name, param_obj):
        name = cls.name
        if param_obj.kind in (Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY):
            raise ValidationError(("Component '{}' __init__ uses non-keyword argument '{}', which is not " +
                                   "supported").format(name, param_name))
        if param_obj.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
            raise ValidationError(("Component '{}' __init__ uses *args or **kwargs, which is not " +
                                   "supported").format(name))
        if param_name in cls._INVALID_SUBCLASS_INIT_ARGS:
            raise ValidationError(("Component '{}' __init__ should not provide argument '{}'").format(name, param_name))
        if param_obj.default == Signature.empty:
            raise ValidationError(("Component '{}' __init__ has no default value for argument '{}'").format(name, param_name))
        return param_obj.default

    @classmethod
    def _introspect_default_parameters(cls):
        """Introspect on subclass __init__ method to determine default values of each argument.

        Raises exception if subclass __init__ uses any args other than standard keyword args.

        Returns:
            dict: map from parameter name to default value
        """
        sig = signature(cls.__init__)
        defaults = {}
        for param_name, param_obj in sig.parameters.items():
            if param_name == 'self':
                continue
            defaults[param_name] = cls._introspect_default_parameter(param_name, param_obj)
        missing_subclass_init_args = set(cls._REQUIRED_SUBCLASS_INIT_ARGS) - defaults.keys()
        if len(missing_subclass_init_args):
            name = cls.name
            raise ValidationError("Component '{}' __init__ missing values for required parameters: '{}'".format(name, str(missing_subclass_init_args)))
        return defaults

    def _introspect_parameters(self):
        """Introspect on subclass __init__ method to determine the values saved as state.

        Raises exception if subclass __init__ uses any args other than standard keyword args.
        Also raises exception if parameters defined in subclass __init__ are different from those which
        were provided to ComponentBase.__init__.

        Returns:
            dict: map from parameter name to default value
        """
        sig = signature(self.__init__)
        defaults = self._introspect_default_parameters()
        values = {}
        for param_name, param_obj in sig.parameters.items():
            defaults[param_name] = self._introspect_default_parameter(param_name, param_obj)
            if param_name not in self._REQUIRED_SUBCLASS_INIT_ARGS and not hasattr(self, param_name):
                name = self.name
                raise ValidationError(("Component '{}' __init__ has not saved state for parameter '{}'").format(name, param_name))
            values[param_name] = getattr(self, param_name)
        return values

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
            logger.log_subtitle(title)
        for parameter in self.parameters:
            parameter_str = ("\t * {} : {}").format(parameter, self.parameters[parameter])
            logger.log(parameter_str)
        if return_dict:
            component_dict = {"name": self.name}
            component_dict.update({"parameters": self.parameters})
            return component_dict
