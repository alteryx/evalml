from .regression_objective import RegressionObjective

from evalml.problem_types import ProblemTypes


class TimeSeriesRegressionObjective(RegressionObjective):
    """Base class for all time series regression objectives."""

    name = "Time Series Regression Objective"
    problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
