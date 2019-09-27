import pickle

from .classification import (
    LogisticRegressionPipeline,
    RFClassificationPipeline,
    XGBoostPipeline
)
from .regression import RFRegressionPipeline

from evalml.problem_types import handle_problem_types

ALL_PIPELINES = [RFClassificationPipeline, XGBoostPipeline, LogisticRegressionPipeline, RFRegressionPipeline]


def get_pipelines(problem_type, model_types=None):
    """Returns potential pipelines by model type

    Arguments:

        problem_type(ProblemTypes or str): the problem type the pipelines work for.
        model_types(list[str]): model types to match. if none, return all pipelines

    Returns

        pipelines, list of all pipeline

    """
    problem_pipelines = []

    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES:
        if problem_type in p.problem_types:
            problem_pipelines.append(p)

    if model_types is None:
        return problem_pipelines

    all_model_types = list_model_types(problem_type)
    for model_type in model_types:
        if model_type not in all_model_types:
            raise RuntimeError("Unrecognized model type for problem type %s: %s f" % (problem_type, model_type))

    pipelines = []

    for p in problem_pipelines:
        if p.model_type in model_types:
            pipelines.append(p)

    return pipelines


def list_model_types(problem_type):
    """List model type for a particular problem type

    Arguments:
        problem_types (ProblemType or str): binary, multiclass, or regression

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
        pickle.dump(pipeline, f)


def load_pipeline(file_path):
    """Loads pipeline at file path

    Args:
        file_path (str) : location to load file

    Returns:
        Pipeline obj
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)
