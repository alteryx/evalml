from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    """
    All regression objectives should inherit from this class.

    problem_type (ProblemTypes): type of problem this objective is. Set to ProblemTypes.REGRESSION.
    """
    problem_type = ProblemTypes.REGRESSION
