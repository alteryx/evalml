from abc import ABC, abstractmethod


class ObjectiveBase(ABC):
    name = None
    problem_type = None

    #  TODO: replace with None and enforce
    greater_is_better = True
    score_needs_proba = False

    def __init__(self):
        if self.name is None:
            raise NameError("Objective `name` cannot be set to None.")
        if self.problem_type is None:
            raise NameError("Objective `problem_type` cannot be set to None.")
        if self.greater_is_better is None:
            raise NameError("Objective `greater_is_better` cannot be set to None.")
        if self.score_needs_proba is None:
            raise NameError("Objective `score_needs_proba` cannot be set to None.")

    @abstractmethod
    def objective_function(self, y_predicted, y_true, X=None):
        raise NotImplementedError("Objective's `objective_function` must be implemented.")

    def score(self, y_predicted, y_true, X=None):
        return self.objective_function(y_predicted, y_true, X=X)
