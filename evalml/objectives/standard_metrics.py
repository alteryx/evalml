from sklearn import metrics

from .objective_base import ObjectiveBase


# todo does this need tuning?
class F1(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "F1"

    def score(self, y_true, y_predicted):
        return metrics.f1_score(y_true, y_predicted)

# todo does this need tuning?


class Precision(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Precision"

    def score(self, y_true, y_predicted):
        return metrics.precision_score(y_true, y_predicted)


class Recall(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "Recall"

    def score(self, y_true, y_predicted):
        return metrics.f1_score(y_true, y_predicted)


class AUC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = True
    name = "AUC"

    def score(self, y_true, y_predicted):
        return metrics.roc_auc_score(y_true, y_predicted)


class LogLoss(ObjectiveBase):
    needs_fitting = False
    greater_is_better = False
    need_proba = True
    name = "Log Loss"

    def score(self, y_true, y_predicted):
        return metrics.log_loss(y_true, y_predicted)


class MCC(ObjectiveBase):
    needs_fitting = False
    greater_is_better = True
    need_proba = False
    name = "MCC"

    def score(self, y_true, y_predicted):
        return metrics.matthews_corrcoef(y_true, y_predicted)
