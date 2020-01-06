import os

import cloudpickle
import yaml

from .components.utils import handle_component

from evalml.model_types import handle_model_types
from evalml.problem_types import handle_problem_types

ALL_PIPELINES = None
pipelines_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_pipelines.yaml")
with open(pipelines_path, 'r') as stream:
    try:
        ALL_PIPELINES = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def register_pipelines(component_lists=None):
    for component_list in component_lists:
        # check model_type
        estimator = handle_component(component_list[-1])
        model_type = estimator.model_type
        problem_types = estimator.problem_types
        for problem_type in problem_types:
            if component_list not in ALL_PIPELINES[problem_type.name][model_type.value]:
                ALL_PIPELINES[problem_type.name][model_type.value].append(component_list)


if 'EVALML_CUSTOM_PIPELINES_PATH' in os.environ:
    with open(os.environ['EVALML_CUSTOM_PIPELINES_PATH']) as stream:
        pipelines = yaml.safe_load(stream)
        register_pipelines(pipelines)


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
    if model_types:
        model_types = [handle_model_types(model_type) for model_type in model_types]
        all_model_types = list_model_types(problem_type)
        for model_type in model_types:
            if model_type not in all_model_types:
                raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_type))

    problem_type = handle_problem_types(problem_type)
    problem_pipelines = ALL_PIPELINES[problem_type.name]
    pipelines = []
    for model_type in problem_pipelines:
        if model_types and handle_model_types(model_type) in model_types:
            for component_list in problem_pipelines[model_type]:
                pipelines.append(component_list)
        elif not model_types:
            for component_list in problem_pipelines[model_type]:
                pipelines.append(component_list)

    return pipelines


def list_model_types(problem_type):
    """List model type for a particular problem type

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        model_types, list of model types
    """

    model_types = []
    problem_type = handle_problem_types(problem_type)
    for p in ALL_PIPELINES[problem_type.name]:
        model_types.append(p)

    return list(set([handle_model_types(p) for p in model_types]))


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
