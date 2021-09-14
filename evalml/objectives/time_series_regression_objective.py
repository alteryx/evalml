"""Base class for all time series regression objectives."""
from .regression_objective import RegressionObjective

from evalml.problem_types import ProblemTypes


class TimeSeriesRegressionObjective(RegressionObjective):
    """Base class for all time series regression objectives."""

    problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
    """[ProblemTypes.TIME_SERIES_REGRESSION]"""
