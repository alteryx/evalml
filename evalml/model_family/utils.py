from .model_family import ModelFamily

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
    from evalml.pipelines import all_estimators

    estimators = []
    problem_type = handle_problem_types(problem_type)
    for estimator in all_estimators():
        if problem_type in [handle_problem_types(pt) for pt in estimator.supported_problem_types]:
            estimators.append(estimator)

    return list(set([e.model_family for e in estimators]))
