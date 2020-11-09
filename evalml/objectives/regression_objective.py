from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    """Base class for all regression objectives.

    problem_types (list(ProblemType)): List of problem types that this objective is defined for.
        Set to [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]
    """

    problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]
    score_needs_proba = False
