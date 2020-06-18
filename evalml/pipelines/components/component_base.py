import copy
from abc import ABC, abstractmethod

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.utils import get_logger, get_random_state, log_subtitle

logger = get_logger(__file__)


class ComponentBase(ABC):
    "Base class for all components"

    def __init__(self, parameters=None, component_obj=None, random_state=0, **kwargs):
        self.random_state = get_random_state(random_state)
        self._component_obj = component_obj
        self._parameters = parameters or {}

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
