from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    name = None
    problem_type = None
    greater_is_better = None
    score_needs_proba = None

    @property
    @staticmethod
    @abstractmethod
    def name(self):
        raise NotImplementedError("This objective must have a `name` attribute as a class variable")

    @property
    @staticmethod
    @abstractmethod
    def problem_type(self):
        raise NotImplementedError("This objective must have a `problem_type` attribute as a class variable")

    @property
    @staticmethod
    @abstractmethod
    def greater_is_better(self):
        raise NotImplementedError("This objective must have a `greater_is_better` boolean attribute as a class variable")

    @property
    @staticmethod
    @abstractmethod
    def score_needs_proba(self):
        raise NotImplementedError("This objective must have a `score_needs_proba` boolean attribute as a class variable")

    @abstractmethod
    def objective_function(self, y_predicted, y_true, X=None):
        raise NotImplementedError("`objective_function` must be implemented.")

    def score(self, y_predicted, y_true, X=None):
        return self.objective_function(y_predicted, y_true, X=X)
