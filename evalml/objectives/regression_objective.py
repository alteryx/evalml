from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class RegressionObjective(ObjectiveBase):
    """Base class for all regression objectives.

    problem_type (ProblemTypes): Type of problem this objective is. Set to ProblemTypes.REGRESSION.
    """

    problem_type = ProblemTypes.REGRESSION
    score_needs_proba = False
