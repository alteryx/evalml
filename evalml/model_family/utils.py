from .model_family import ModelFamily

from evalml.pipelines.components.utils import _all_estimators_used_in_search
from evalml.problem_types import handle_problem_types


def handle_model_family(model_family):
    """Handles model_family by either returning the ModelFamily or converting from a str
    Args:
        model_family (str or ModelFamily) : model type that needs to be handled
    Returns:
        ModelFamily
    """

    if isinstance(model_family, str):
        try:
            tpe = ModelFamily[model_family.upper()]
            return tpe
        except KeyError:
            raise KeyError('Model family \'{}\' does not exist'.format(model_family))
    if isinstance(model_family, ModelFamily):
        return model_family
    raise ValueError('`handle_model_family` was not passed a str or ModelFamily object')


def list_model_families(problem_type):
    """List model type for a particular problem type.

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        list[ModelFamily]: a list of model families
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in _all_estimators_used_in_search:
        if problem_type in set(handle_problem_types(problem) for problem in p.supported_problem_types):
            problem_pipelines.append(p)

    return list(set([p.model_family for p in problem_pipelines]))
