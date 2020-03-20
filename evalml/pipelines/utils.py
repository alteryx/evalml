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

from evalml.model_family import handle_model_family
from evalml.problem_types import handle_problem_types

ALL_PIPELINES = [RFClassificationPipeline,
                 XGBoostPipeline,
                 LogisticRegressionPipeline,
                 LinearRegressionPipeline,
                 RFRegressionPipeline,
                 CatBoostClassificationPipeline,
                 CatBoostRegressionPipeline]


def get_pipelines(problem_type, model_families=None):
    """Returns potential pipelines by model type

    Args:
        problem_type(ProblemTypes or str): the problem type the pipelines work for.
        model_families(list[ModelFamily or str]): model types to match. if none, return all pipelines

    Returns:
        pipelines, list of all pipeline
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")

    problem_pipelines = []

    if model_families:
        model_families = [handle_model_family(model_family) for model_family in model_families]

    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES:
        problem_types = [handle_problem_types(pt) for pt in p.supported_problem_types]
        if problem_type in problem_types:
            problem_pipelines.append(p)

    if model_families is None:
        return problem_pipelines

    all_model_families = list_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    pipelines = []

    for p in problem_pipelines:
        if p.model_family in model_families:
            pipelines.append(p)

    return pipelines


def list_model_families(problem_type):
    """List model type for a particular problem type

    Args:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        model_families, list of model families
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES:
        problem_types = [handle_problem_types(pt) for pt in p.supported_problem_types]
        if problem_type in problem_types:
            problem_pipelines.append(p)

    return list(set([p.model_family for p in problem_pipelines]))


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
