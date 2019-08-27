import numpy as np
from sklearn import metrics

from .objective_base import ObjectiveBase


# todo does this need tuning?
class F1(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1"
    problem_types = ['binary']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted)

class F1Micro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1_Micro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='micro')


class F1Macro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1_Macro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='macro')


class F1Weighted(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1_Weighted"
    problem_types = ['multiclass']

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='weighted')


class Precision(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision"
    problem_types = ['binary']

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted)


class PrecisionMicro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision_Micro"
    problem_types = ['multiclass']

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='micro')


class PrecisionMacro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision_Macro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='macro')


class PrecisionWeighted(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision_Weighted"
    problem_types = ['multiclass']

    def __init__(self, average='binary'):
        self.average = average

    def score(self, y_predicted, y_true):
        return metrics.precision_score(y_true, y_predicted, average='weighted')


class Recall(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall"
    problem_types = ['binary']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted)


class RecallMicro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall_Micro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='micro')


class RecallMacro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall_Macro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='macro')


class RecallWeighted(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        return metrics.f1_score(y_true, y_predicted, average='weighted')


class AUC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC"
    problem_types = ['binary']

    def score(self, y_predicted, y_true):
        return metrics.roc_auc_score(y_true, y_predicted)


class AUCMicro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC_Micro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='micro')


class AUCMacro(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC_Macro"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='macro')


class AUCWeighted(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    score_needs_proba = True
    name = "AUC"
    problem_types = ['multiclass']

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.roc_auc_score(y_true, y_predicted, average='weighted')


class LogLoss(ObjectiveBase):
    needs_fitting = False
    greater_is_better = False
    score_needs_proba = True
    name = "Log Loss"
    problem_types = ['binary', 'multiclass']

    def score(self, y_predicted, y_true):
        y_true, y_predicted = _handle_predictions(y_true, y_predicted)
        return metrics.log_loss(y_true, y_predicted)


class MCC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "MCC"
    problem_types = ['binary', 'multiclass']

    def score(self, y_predicted, y_true):
        return metrics.matthews_corrcoef(y_true, y_predicted)


class R2(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "R2"
    problem_types = ['regression']

    def score(self, y_predicted, y_true):
        return metrics.r2_score(y_true, y_predicted)


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)
        y_pred = label_binarize(y_pred, classes=classes)

    return y_true, y_pred
