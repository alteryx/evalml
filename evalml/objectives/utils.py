"""Utility methods for EvalML objectives."""
from .objective_base import ObjectiveBase

from evalml import objectives
from evalml.exceptions import ObjectiveCreationError, ObjectiveNotFoundError
from evalml.problem_types import handle_problem_types
from evalml.utils.gen_utils import _get_subclasses


def get_non_core_objectives():
    """Get non-core objective classes.

    Non-core objectives are objectives that are domain-specific. Users typically need to configure these objectives
    before using them in AutoMLSearch.

    Returns:
        List of ObjectiveBase classes
    """
    return [
        objectives.CostBenefitMatrix,
        objectives.FraudCost,
        objectives.LeadScoring,
        objectives.Recall,
        objectives.RecallMacro,
        objectives.RecallMicro,
        objectives.RecallWeighted,
        objectives.MAPE,
        objectives.MeanSquaredLogError,
        objectives.RootMeanSquaredLogError,
        objectives.SensitivityLowAlert,
    ]


def _all_objectives_dict():
    all_objectives = _get_subclasses(ObjectiveBase)
    objectives_dict = {}
    for objective in all_objectives:
        if "evalml.objectives" not in objective.__module__:
            continue
        objectives_dict[objective.name.lower()] = objective
    return objectives_dict


def get_all_objective_names():
    """Get a list of the names of all objectives.

    Returns:
        list (str): Objective names
    """
    all_objectives_dict = _all_objectives_dict()
    return list(all_objectives_dict.keys())


def get_core_objective_names():
    """Get a list of all valid core objectives.

    Returns:
        list[str]: Objective names.
    """
    all_objectives = _all_objectives_dict()
    non_core = get_non_core_objectives()
    return [
        name
        for name, class_name in all_objectives.items()
        if class_name not in non_core
    ]


def get_objective(objective, return_instance=False, **kwargs):
    """Returns the Objective class corresponding to a given objective name.

    Args:
        objective (str or ObjectiveBase): Name or instance of the objective class.
        return_instance (bool): Whether to return an instance of the objective. This only applies if objective
            is of type str. Note that the instance will be initialized with default arguments.
        kwargs (Any): Any keyword arguments to pass into the objective. Only used when return_instance=True.

    Returns:
        ObjectiveBase if the parameter objective is of type ObjectiveBase. If objective is instead a valid
        objective name, function will return the class corresponding to that name. If return_instance is True,
        an instance of that objective will be returned.

    Raises:
        TypeError: If objective is None.
        TypeError: If objective is not a string and not an instance of ObjectiveBase.
        ObjectiveNotFoundError: If input objective is not a valid objective.
        ObjectiveCreationError: If objective cannot be created properly.
    """
    if objective is None:
        raise TypeError("Objective parameter cannot be NoneType")
    if isinstance(objective, ObjectiveBase):
        return objective
    all_objectives_dict = _all_objectives_dict()
    if not isinstance(objective, str):
        raise TypeError(
            "If parameter objective is not a string, it must be an instance of ObjectiveBase!"
        )
    if objective.lower() not in all_objectives_dict:
        raise ObjectiveNotFoundError(
            f"{objective} is not a valid Objective! "
            "Use evalml.objectives.get_all_objective_names()"
            "to get a list of all valid objective names. "
        )

    objective_class = all_objectives_dict[objective.lower()]

    if return_instance:
        try:
            return objective_class(**kwargs)
        except TypeError as e:
            raise ObjectiveCreationError(
                f"In get_objective, cannot pass in return_instance=True for {objective} because {str(e)}"
            )

    return objective_class


def get_core_objectives(problem_type):
    """Returns all core objective instances associated with the given problem type.

    Core objectives are designed to work out-of-the-box for any dataset.

    Args:
        problem_type (str/ProblemTypes): Type of problem

    Returns:
        List of ObjectiveBase instances
    """
    problem_type = handle_problem_types(problem_type)
    all_objectives_dict = _all_objectives_dict()
    objectives = [
        obj()
        for obj in all_objectives_dict.values()
        if obj.is_defined_for_problem_type(problem_type)
        and obj not in get_non_core_objectives()
    ]
    return objectives
