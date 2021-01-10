from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MulticlassClassificationObjective(ObjectiveBase):
    """Base class for all multiclass classification objectives."""

    name = "Multiclass Classification Objective"
    problem_types = [ProblemTypes.MULTICLASS, ProblemTypes.TIME_SERIES_MULTICLASS]
