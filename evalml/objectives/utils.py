import numpy as np
from sklearn.preprocessing import label_binarize

from . import standard_metrics
from .objective_base import ObjectiveBase


def get_objective(objective):
    if isinstance(objective, ObjectiveBase):
        return objective
    objective = objective.lower()
    extra_param = None

    options = {
        "f1": standard_metrics.F1(),
        'f1_micro': standard_metrics.F1Micro(),
        'f1_macro': standard_metrics.F1Macro(),
        'f1_weighted': standard_metrics.F1Weighted(),
        "precision": standard_metrics.Precision(),
        "precision_micro": standard_metrics.PrecisionMicro(),
        "precision_macro": standard_metrics.PrecisionMacro(),
        "precision_weighted": standard_metrics.PrecisionWeighted(),
        "recall": standard_metrics.Recall(),
        "recall_micro": standard_metrics.RecallMicro(),
        "recall_macro": standard_metrics.RecallMacro(),
        "recall_weighted": standard_metrics.RecallWeighted(),
        "auc": standard_metrics.AUC(),
        "auc_micro": standard_metrics.AUCMicro(),
        "auc_macro": standard_metrics.AUCMacro(),
        "auc_weighted": standard_metrics.AUCWeighted(),
        "log_loss": standard_metrics.LogLoss(),
        "mcc": standard_metrics.MCC(),
        "r2": standard_metrics.R2(),
    }

    return options[objective]


def _handle_predictions(y_true, y_pred):
    if len(np.unique(y_true)) > 2:
        classes = np.unique(y_true)
        y_true = label_binarize(y_true, classes=classes)
        y_pred = label_binarize(y_pred, classes=classes)

    return y_true, y_pred
