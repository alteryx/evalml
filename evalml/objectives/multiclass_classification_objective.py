from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MulticlassClassificationObjective(ObjectiveBase):
    """
    Base class for all multiclass classification objectives.

    problem_type (ProblemTypes): Specifies the type of problem this objective is defined for (multiclass classification).
    """
    problem_type = ProblemTypes.MULTICLASS
