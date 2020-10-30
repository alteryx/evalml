from enum import Enum

from evalml.utils import classproperty


class ProblemTypes(Enum):
    """Enum defining the supported types of machine learning problems."""
    BINARY = 'binary'
    """Binary classification problem."""
    MULTICLASS = 'multiclass'
    """Multiclass classification problem."""
    REGRESSION = 'regression'
    """Regression problem."""
    TIME_SERIES_REGRESSION = 'time_series_regression'
    """Time series regression problem."""

    def __str__(self):
        problem_type_dict = {ProblemTypes.BINARY.name: "binary",
                             ProblemTypes.MULTICLASS.name: "multiclass",
                             ProblemTypes.REGRESSION.name: "regression",
                             ProblemTypes.TIME_SERIES_REGRESSION.name: "time_series_regression"}
        return problem_type_dict[self.name]

    @classproperty
    def all_problem_types(cls):
        """Get a list of all defined problem types.

        Returns:
            list(ProblemTypes): list
        """
        return list(cls)
