from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ObjectiveBase):
    """
    Base class for all multi-class classification objectives.

    problem_type (ProblemTypes): Specifies the type of problem this objective is defined for (multiclass classification).
    """
    problem_type = ProblemTypes.MULTICLASS
