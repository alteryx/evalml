from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MulticlassClassificationObjective(ObjectiveBase):
    """Base class for all multiclass classification objectives.

    problem_type (ProblemTypes): Type of problem this objective is. Set to ProblemTypes.MULTICLASS.
    """
    problem_type = ProblemTypes.MULTICLASS
