import numpy as np

from .objective_base import ObjectiveBase

from evalml.problem_types import ProblemTypes


class MultiClassificationObjective(ObjectiveBase):
    problem_type = ProblemTypes.MULTICLASS

    def decision_function(self, ypred_proba, threshold=None, X=None):
        #  we don't have anything that actually calls this right now i think?
        return np.argmax(ypred_proba)
