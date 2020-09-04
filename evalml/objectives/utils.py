from texttable import Texttable

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
        objectives_dict[objective.name.lower()] = objective
    return objectives_dict


def _print_objectives_in_table(names):
    """Print the list of objective names in a table.

    Returns:
         None
    """
    def iterate_in_batches(sequence, batch_size):
        return [sequence[pos:pos + batch_size] for pos in range(0, len(sequence), batch_size)]
    batch_size = 4
    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HLINES | Texttable.VLINES)
    for row in iterate_in_batches(sorted(names), batch_size):
        if len(row) < batch_size:
            row += [""] * (batch_size - len(row))
        table.add_row(row)
    print(table.draw())


def print_all_objective_names():
    """Get all valid objective names in a table.

    Returns:
        None
    """
    all_objectives_dict = _all_objectives_dict()
    all_names = list(all_objectives_dict.keys())
    _print_objectives_in_table(all_names)


def get_objective(objective, return_instance=False, **kwargs):
    """Returns the Objective object of the given objective name

    Arguments:
        objective (str or ObjectiveBase): name or instance of the objective class.
        return_instance (bool): Whether to return an instance of the objective. This only applies if objective
            is of type str. Note that the instance will be initialized with default arguments.
        **kwargs (Any): Any keyword arguments to pass into the objective. Only used when return_instance=True.

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
    if not isinstance(objective, str):
        raise TypeError("If parameter objective is not a string, it must be an instance of ObjectiveBase!")
    if objective.lower() not in all_objectives_dict:
        raise ObjectiveNotFoundError(f"{objective} is not a valid Objective! "
                                     "Use evalml.objectives.print_all_objective_names(allowed_in_automl=False)"
                                     "to get a list of all valid objective names. ")

    objective_class = all_objectives_dict[objective.lower()]

    if return_instance:
        try:
            return objective_class(**kwargs)
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
    objectives = [obj for obj in all_objectives_dict.values() if obj.problem_type == problem_type]
    return objectives
