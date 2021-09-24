"""Base class for all clustering objectives."""
import numpy as np

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class ClusteringObjective(ObjectiveBase):
    """Base class for all clustering objectives."""

    problem_types = [ProblemTypes.CLUSTERING]
    """[ProblemTypes.CLUSTERING]"""
