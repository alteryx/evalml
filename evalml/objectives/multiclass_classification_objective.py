import numpy as np

from .objective_base import ObjectiveBase


class MultiClassificationObjective(ObjectiveBase):
    can_optimize_bin_class_threshold = False

    # def decision_function(self, ypred_proba, X=None):
    #     return np.argmax(ypred_proba)
