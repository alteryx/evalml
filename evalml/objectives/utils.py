from . import standard_metrics
from .objective_base import ObjectiveBase

OPTIONS = {
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


def get_objective(objective):
    """Returns the Objective object of the given objective name

    Args:
        objective (str) : name of the objective

    Returns:
        Objective
    """
    if isinstance(objective, ObjectiveBase):
        return objective
    objective = objective.lower()
    return OPTIONS[objective]


def get_objectives(objective_type):
    """Returns all objectives associated with the given problem type

    Args:
        problem_type (str) : type of machine learning problem

    Returns:
        List of Objectives
    """
    return [obj for obj in OPTIONS if objective_type in OPTIONS[obj].objective_types]
