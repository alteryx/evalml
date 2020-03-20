from .classification_objective import ClassificationObjective

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ClassificationObjective):
    """
    Base class for all multi-class classification objectives.

    problem_type (ProblemTypes): Type of problem this objective is. Set to ProblemTypes.MULTICLASS.
    """
    problem_type = ProblemTypes.MULTICLASS

    def decision_function(self, ypred_proba, threshold=None, X=None):
        raise NotImplementedError("decision_function for a multiclass classification objective is not yet defined!")
