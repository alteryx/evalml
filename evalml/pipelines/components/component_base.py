from evalml.utils import Logger


class ComponentBase:
    def __init__(self, parameters, component_obj, random_state):
        self.random_state = random_state
        self._component_obj = component_obj
        self.parameters = parameters
        self.logger = Logger()

        attributes_to_check = ['_needs_fitting', "name", "component_type"]

        for attribute in attributes_to_check:
            if not hasattr(self, attribute):
                raise AttributeError("Component missing attribute: `{}`".format(attribute))

    def fit(self, X, y=None):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self
        """
        try:
            return self._component_obj.fit(X, y)
        except AttributeError:
            raise RuntimeError("Component requires a fit method or a component_obj that implements fit")

    def describe(self, print_name=False, return_dict=False):
        """Describe a component and its parameters
        """
        if print_name:
            title = self.name
            self.logger.log_subtitle(title)
        for parameter in self.parameters:
            parameter_str = ("\t * {} : {}").format(parameter, self.parameters[parameter])
            self.logger.log(parameter_str)
        if return_dict:
            component_dict = {}
            component_dict.update({"name": self.name})
            component_dict.update({"parameters": self.parameters})
            return component_dict
