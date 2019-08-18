from .objective_base import ObjectiveBase
from .standard_metrics import Precision


def get_objective(objective):
    if isinstance(objective, ObjectiveBase):
        return objective

    options = {
        "precision": Precision()
    }

    return options[objective]
