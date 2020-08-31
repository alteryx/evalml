import texttable

from .objective_base import ObjectiveBase

from evalml.exceptions import ObjectiveNotFoundError
from evalml.problem_types import handle_problem_types
from evalml.utils.gen_utils import _get_subclasses


def _all_objectives_dict():
    all_objectives = _get_subclasses(ObjectiveBase)
    objectives_dict = {}
    for objective in all_objectives:
        if 'evalml.objectives' not in objective.__module__:
            continue
        objectives_dict[objective.name] = objective
        objectives_dict[objective.name.lower()] = objective
    return objectives_dict


def iterate_in_batches(sequence, batch_size):
    return [sequence[pos:pos + batch_size] for pos in range(0, len(sequence), batch_size)]


def pretty_print_all_valid_objective_names():
    all_objectives_dict = _all_objectives_dict()
    table = texttable.Texttable()
    return table.add_rows(iterate_in_batches(sorted(list(all_objectives_dict.keys())), 4)).draw()


def get_objective(objective, return_instance=False):
    """Returns the Objective object of the given objective name

    Args:
        objective (str or ObjectiveBase): name or instance of the objective class.
        return_instance (bool): Whether to return an instance of the objective. This only applies if objective
            is of type str. Note that the instance will be initialized with default arguments.

    Returns:
        ObjectiveBase if the parameter objective is of type ObjectiveBase. If objective is instead a valid
        objective name, function will return the class corresponding to that name. If return_instance is True,
        an instance of that objective will be returned.
    """
    if objective is None:
        raise TypeError("Objective parameter cannot be NoneType")
    if isinstance(objective, ObjectiveBase):
        return objective
    all_objectives_dict = _all_objectives_dict()
    if objective not in all_objectives_dict:
        valid_names = pretty_print_all_valid_objective_names()
        raise ObjectiveNotFoundError(f"{objective} is not a valid Objective! " +
                                     "Objective must be one of:\n" +
                                     valid_names)

    objective_class = all_objectives_dict[objective]

    if return_instance:
        try:
            return objective_class()
        except TypeError as e:
            raise TypeError(f"In get_objective, cannot pass in return_instance=True for {objective} because {str(e)}")

    return objective_class


def get_objectives(problem_type):
    """Returns all objective classes associated with the given problem type.

    Args:
        problem_type (str/ProblemTypes): type of problem

    Returns:
        List of Objectives
    """
    problem_type = handle_problem_types(problem_type)
    all_objectives_dict = _all_objectives_dict()
    # To remove duplicates
    objectives = [obj for name, obj in all_objectives_dict.items() if obj.problem_type == problem_type and obj.name == name]
    return objectives
