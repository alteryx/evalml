from evalml.exceptions import MethodPropertyNotFoundError
from evalml.utils import Logger

logger = Logger()


class ComponentBase:
    def __init__(self, parameters, component_obj, random_state):
        self.random_state = random_state
        self._component_obj = component_obj
        self.parameters = parameters

        attributes_to_check = ["name"]

        for attribute in attributes_to_check:
            if not hasattr(self, attribute):
                raise AttributeError("Component missing attribute: `{}`".format(attribute))

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
