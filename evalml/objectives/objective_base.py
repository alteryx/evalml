from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    """Base class for all objectives."""
    name = None
    problem_type = None
    greater_is_better = None
    score_needs_proba = None

    @property
    @classmethod
    @abstractmethod
    def name(self):
        """Returns a name describing the objective."""
        raise NotImplementedError("This objective must have a `name` attribute as a class variable")

    @property
    @classmethod
    @abstractmethod
    def problem_type(self):
        """Returns a ProblemTypes enum describing the problem type the objective can handle."""
        raise NotImplementedError("This objective must have a `problem_type` attribute as a class variable")

    @property
    @classmethod
    @abstractmethod
    def greater_is_better(self):
        """Returns a boolean determining if a greater score indicates better model performance."""
        raise NotImplementedError("This objective must have a `greater_is_better` boolean attribute as a class variable")

    @property
    @classmethod
    @abstractmethod
    def score_needs_proba(self):
        """Returns a boolean determining if `score()` needs probability estimates."""
        raise NotImplementedError("This objective must have a `score_needs_proba` boolean attribute as a class variable")

    @classmethod
    @abstractmethod
    def objective_function(self, y_predicted, y_true, X=None):
        """Computes the relative value of the provided predictions compared to the true values, according a specified metric
         Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : true values of length [n_samples]
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            numerical value used to calculate score
        """
        raise NotImplementedError("`objective_function` must be implemented.")

    def score(self, y_predicted, y_true, X=None):
        """Returns a numerical score indicating performance based on the differences between the predicted and actual values.

        Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : true values of length [n_samples]
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            score
        """
        return self.objective_function(y_predicted, y_true, X=X)
