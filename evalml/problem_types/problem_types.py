"""Enum defining the supported types of machine learning problems."""
from enum import Enum

from evalml.utils import classproperty


class ProblemTypes(Enum):
    """Enum defining the supported types of machine learning problems."""

    BINARY = "binary"
    """Binary classification problem."""
    MULTICLASS = "multiclass"
    """Multiclass classification problem."""
    REGRESSION = "regression"
    """Regression problem."""
    TIME_SERIES_REGRESSION = "time series regression"
    """Time series regression problem."""
    TIME_SERIES_BINARY = "time series binary"
    """Time series binary classification problem."""
    TIME_SERIES_MULTICLASS = "time series multiclass"
    """Time series multiclass classification problem."""

    def __str__(self):
        """String representation of the ProblemTypes enum."""
        problem_type_dict = {
            ProblemTypes.BINARY.name: "binary",
            ProblemTypes.MULTICLASS.name: "multiclass",
            ProblemTypes.REGRESSION.name: "regression",
            ProblemTypes.TIME_SERIES_REGRESSION.name: "time series regression",
            ProblemTypes.TIME_SERIES_BINARY.name: "time series binary",
            ProblemTypes.TIME_SERIES_MULTICLASS.name: "time series multiclass",
        }
        return problem_type_dict[self.name]

    @classproperty
    def _all_values(cls):
        return {pt.value.upper(): pt for pt in cls.all_problem_types}

    @classproperty
    def all_problem_types(cls):
        """Get a list of all defined problem types.

        Returns:
            list(ProblemTypes): List of all defined problem types.
        """
        return list(cls)
