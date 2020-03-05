from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    problem_type = ProblemTypes.REGRESSION
