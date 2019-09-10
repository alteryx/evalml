from . import standard_metrics
from .objective_base import ObjectiveBase

from evalml.problem_types import handle_problem_types

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


def get_objectives(problem_type):
    """Returns all objectives associated with the given problem types

    Args:
        problem_type (str/ProblemTypes) : type of problem

    Returns:
        List of Objectives
    """
    problem_type = handle_problem_types(problem_type)
    return [obj for obj in OPTIONS if OPTIONS[obj].supports_problem_type(problem_type)]
