from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ObjectiveBase):
    problem_type = ProblemTypes.MULTICLASS

    def decision_function(self, ypred_proba, threshold=None, X=None):
        raise NotImplementedError("decision_function for a multiclass classification objective is not yet defined!")
