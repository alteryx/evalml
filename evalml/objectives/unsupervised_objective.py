"""Base class for all unsupervised learning objectives."""
from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class UnsupervisedLearningObjective(ObjectiveBase):
    """Base class for all unsupervised learning objectives."""

    problem_types = [ProblemTypes.CLUSTERING]
    """[ProblemTypes.CLUSTERING]"""
