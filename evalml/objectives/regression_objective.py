"""Base class for all regression objectives."""
from evalml.objectives.objective_base import ObjectiveBase
from evalml.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    """Base class for all regression objectives."""

    problem_types = [ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.REGRESSION, ProblemTypes.TIME_SERIES_REGRESSION]"""
