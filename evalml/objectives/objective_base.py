from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    """Base class for all objectives."""

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns a name describing the objective."""
        raise NotImplementedError("This objective must have a `name` attribute as a class variable")

    @property
    @classmethod
    @abstractmethod
    def greater_is_better(cls):
        """Returns a boolean determining if a greater score indicates better model performance."""
        raise NotImplementedError("This objective must have a `greater_is_better` boolean attribute as a class variable")

    @property
    @classmethod
    @abstractmethod
    def score_needs_proba(cls):
        """Returns a boolean determining if `score()` needs probability estimates."""
        raise NotImplementedError("This objective must have a `score_needs_proba` boolean attribute as a class variable")

    @classmethod
    @abstractmethod
    def objective_function(cls, y_predicted, y_true, X=None):
        """Computes the relative value of the provided predictions compared to the actual labels, according a specified metric
         Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : actual class labels of length [n_samples]
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            numerical value used to calculate score
        """
        raise NotImplementedError("`objective_function` must be implemented.")

    def score(self, y_predicted, y_true, X=None):
        """Returns a numerical score indicating performance based on the differences between the predicted and actual values.

        Arguments:
            y_predicted (pd.Series) : predicted values of length [n_samples]
            y_true (pd.Series) : actual class labels of length [n_samples] 
            X (pd.DataFrame or np.array) : extra data of shape [n_samples, n_features] necessary to calculate score

        Returns:
            score
        """
        return self.objective_function(y_predicted, y_true, X=X)
