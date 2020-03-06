# import numpy as np

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ObjectiveBase):
    can_optimize_threshold = False
    problem_type = ProblemTypes.MULTICLASS
    # def decision_function(self, ypred_proba, X=None):
    #     return np.argmax(ypred_proba)
