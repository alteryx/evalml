import numpy as np
from sklearn import metrics

from .objective_base import ObjectiveBase
from .utils import binarize_y


# todo does this need tuning?
class F1(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1"

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average=self.average)
# todo does this need tuning?


class Precision(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision"

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average=self.average)


class Recall(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall"

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average=self.average)


class AUC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = True
    name = "AUC"

    def __init__(self, average=None):
        self.average = average

    def score(self, y_predicted, y_true):
        if len(np.unique(y_true)) > 2:
            y_true, y_predicted = binarize_y(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average=self.average)


class LogLoss(ObjectiveBase):
    needs_fitting = False
    greater_is_better = False
    need_proba = True
    name = "Log Loss"

    def score(self, y_predicted, y_true):
        if len(np.unique(y_true)) > 2:
            y_true, y_predicted = binarize_y(y_true, y_predicted)
        return metrics.log_loss(y_true, y_predicted)


class MCC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "MCC"

    def score(self, y_predicted, y_true):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "R2"

    def score(self, y_predicted, y_true):
        return metrics.r2_score(y_true, y_predicted)
