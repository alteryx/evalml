from sklearn.metrics import precision_score

from .objective_base import ObjectiveBase


class Precision(ObjectiveBase):
    needs_fitting = False
    greater_is_better = False
    need_proba = False

    def score(self, y_true, y_predicted):
        return precision_score(y_true, y_predicted)
