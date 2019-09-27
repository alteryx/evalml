class ComponentBase:
    def __init__(self, name, component_type, needs_fitting=False, component_obj=None, random_state=0):
        self.name = name
        self.component_type = component_type
        self._needs_fitting = needs_fitting
        self._component_obj = component_obj
        self.random_state = random_state

    def fit(self, X, y, objective_fit_size=.2):
        """Build a model

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

        Returns:
            self
        """
        self._component_obj.fit(X, y)

    def predict(self, X):
        """Make predictions using selected features.

        Args:
            X (DataFrame) : features

        Returns:
            Series : estimated labels
        """
        self._component_obj.predict(X)

    def predict_proba(self, X):
        """Make probability estimates for labels.

        Args:
            X (DataFrame) : features

        Returns:
            DataFrame : probability estimates
        """
        self._component_obj.predict_proba(X)

    def score(self, X, y, other_objectives=None):
        """Evaluate model performance

        Args:
            X (DataFrame) : features for model predictions
            y (Series) : true labels
            other_objectives (list): list of other objectives to score

        Returns:
            score, dictionary of other objective scores
        """
        self._component_obj.score(X, y)
