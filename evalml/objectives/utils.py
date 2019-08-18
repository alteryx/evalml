from . import standard_metrics
from .objective_base import ObjectiveBase


def get_objective(objective):
    if isinstance(objective, ObjectiveBase):
        return objective

    options = {
        "fl": standard_metrics.F1(),
        "precision": standard_metrics.Precision(),
        "recall": standard_metrics.Recall(),
        "auc": standard_metrics.AUC(),
        "log_loss": standard_metrics.LogLoss(),
        "mcc": standard_metrics.MCC(),
    }

    return options[objective]
