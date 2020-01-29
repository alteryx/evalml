import cloudpickle

from .classification import (
    CatBoostClassificationPipeline,
    LogisticRegressionPipeline,
    RFClassificationPipeline,
    XGBoostPipeline
)
from .regression import (
    CatBoostRegressionPipeline,
    LinearRegressionPipeline,
    RFRegressionPipeline
)

from evalml.model_types import handle_model_types
from evalml.problem_types import handle_problem_types
from evalml.utils import import_or_raise

ALL_PIPELINES = [RFClassificationPipeline,
                 XGBoostPipeline,
                 LogisticRegressionPipeline,
                 LinearRegressionPipeline,
                 RFRegressionPipeline,
                 CatBoostClassificationPipeline,
                 CatBoostRegressionPipeline]


def get_pipelines(problem_type, model_types=None):
    """Returns potential pipelines by model type

    Arguments:

        problem_type(ProblemTypes or str): the problem type the pipelines work for.
        model_types(list[ModelTypes or str]): model types to match. if none, return all pipelines

    Returns:
        pipelines, list of all pipeline

    """
    if model_types is not None and not isinstance(model_types, list):
        raise TypeError("model_types parameter is not a list.")

    problem_pipelines = []

    if model_types:
        model_types = [handle_model_types(model_type) for model_type in model_types]

    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES:
        if problem_type in p.problem_types:
            problem_pipelines.append(p)

    if model_types is None:
        return problem_pipelines

    all_model_types = list_model_types(problem_type)
    for model_type in model_types:
        if model_type not in all_model_types:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_type))

    pipelines = []

    for p in problem_pipelines:
        if p.model_type in model_types:
            pipelines.append(p)

    return pipelines


def list_model_types(problem_type):
    """List model type for a particular problem type

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        model_types, list of model types
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES:
        if problem_type in p.problem_types:
            problem_pipelines.append(p)

    return list(set([p.model_type for p in problem_pipelines]))


def save_pipeline(pipeline, file_path):
    """Saves pipeline at file path

    Args:
        file_path (str) : location to save file

    Returns:
        None
    """
    with open(file_path, 'wb') as f:
        cloudpickle.dump(pipeline, f)


def load_pipeline(file_path):
    """Loads pipeline at file path

    Args:
        file_path (str) : location to load file

    Returns:
        Pipeline obj
    """
    with open(file_path, 'rb') as f:
        return cloudpickle.load(f)
