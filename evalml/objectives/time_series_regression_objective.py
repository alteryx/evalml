from .regression_objective import RegressionObjective

from evalml.problem_types import ProblemTypes


class TimeSeriesRegressionObjective(RegressionObjective):
    """Base class for all time series regression objectives.

    problem_types (list(ProblemType)): List of problem types that this objective is defined for.
        Set to [ProblemTypes.TIME_SERIES_REGRESSION]
    """

    problem_types = [ProblemTypes.TIME_SERIES_REGRESSION]
