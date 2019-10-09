class ComponentBase:
    def __init__(self, name, component_type, hyperparameters={}, parameters={}, needs_fitting=False, component_obj=None, random_state=0):
        self.name = name
        self.component_type = component_type
        self.hyperparameters = hyperparameters
        self.random_state = random_state
        self._needs_fitting = needs_fitting
        self._component_obj = component_obj
        self.parameters = parameters
        # self.validate_parameters()

    # TODO need to fix with new hyper-parameter
    # def validate_parameters(self):
    #     for parameter_name in hyperparameters:
    #         if hyperparameters[parameter_name] not in potential_parameters:
    #             raise ValueError("Paremeter {}: is not valid".format(parameter_name))
    #         elif hyperparameters[parameter_name] not in potential_parameters[parameter_name]:
    #             raise valueError("Value {} is not valid for paramater {}".format(hyperparameters[parameter_name]), parameter_name)

    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self
        """
        return self._component_obj.fit(X, y)

    def describe(self, return_dict=False):
        # (ideally could use _log in the future)
        title = self.name
        print(title)
        print("-" * len(title))
        for parameter in self.parameters:
            print("* ", parameter, ":", self.parameters[parameter])
        print("\n")
        if return_dict:
            return self.parameters
