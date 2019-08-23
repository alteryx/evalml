import numpy as np
from sklearn.preprocessing import label_binarize

from . import standard_metrics
from .objective_base import ObjectiveBase


def get_objective(objective):
    if isinstance(objective, ObjectiveBase):
        return objective

    objective = objective.lower()

    options = {
        "f1": standard_metrics.F1(),
        "precision": standard_metrics.Precision(),
        "recall": standard_metrics.Recall(),
        "auc": standard_metrics.AUC(),
        "log_loss": standard_metrics.LogLoss(),
        "mcc": standard_metrics.MCC(),
        "r2": standard_metrics.R2(),
    }

    return options[objective]


def binarize_y(y_true, y_pred):
    classes = np.unique(y_true)
    y_true = label_binarize(y_true, classes=classes)
    y_pred = label_binarize(y_pred, classes=classes)

    return y_true, y_pred
