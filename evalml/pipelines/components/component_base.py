class ComponentBase:
    def __init__(self, name, component_type, hyperparameters, needs_fitting=False, component_obj=None):
        self.name = name
        self.component_type = component_type
        self.potential_parameters = potential_parameters
        self.hyperparameters = hyperparameters
        self._needs_fitting = needs_fitting
        self._component_obj = component_obj
        self.validate_parameters()

    def validate_parameters(self):
        for parameter_name in hyperparameters:
            if hyperparameters[parameter_name] not in potential_parameters:
                raise ValueError("Paremeter {}: is not valid".format(parameter_name))
            elif hyperparameters[parameter_name] not in potential_parameters[parameter_name]:
                raise valueError("Value {} is not valid for paramater {}".format(hyperparameters[parameter_name]), parameter_name)


    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

        Returns:

            self

        """
        pass

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        pass

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """
        pass

    def score(self, X, y, other_objectives=None):
        """Evaluate model performance

        Args:
            X (DataFrame) : features for model predictions
            y (Series) : true labels
            other_objectives (list): list of other objectives to score

        Returns:
            score, dictionary of other objective scores
        """
        pass
