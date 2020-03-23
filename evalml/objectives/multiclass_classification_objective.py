from .classification_objective import ClassificationObjective

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ClassificationObjective):
    """
    Base class for all multi-class classification objectives.

    problem_type (ProblemTypes): Type of problem this objective is. Set to ProblemTypes.MULTICLASS.
    """
    problem_type = ProblemTypes.MULTICLASS
