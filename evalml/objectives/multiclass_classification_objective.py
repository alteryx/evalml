from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MulticlassClassificationObjective(ObjectiveBase):
    """Base class for all multiclass classification objectives.

    problem_types (list(ProblemType)): List of problem types that this objective is defined for.
        Set to [ProblemTypes.MULTICLASS].
    """

    problem_types = [ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]
