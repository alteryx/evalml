from .classification import (
    LogisticRegressionPipeline,
    RFClassificationPipeline,
    XGBoostPipeline
)
from .regression import RFRegressionPipeline

ALL_PIPELINES = [RFClassificationPipeline, XGBoostPipeline, LogisticRegressionPipeline, RFRegressionPipeline]


def get_pipelines(problem_type, model_types=None):
    """Returns potential pipelines by model type

    Arguments:

        problem_type (str): the problem type the pipelines work for. Either regression or classification
        model_types(list[str]): model types to match. if none, return all pipelines

    Returns

        pipelines, list of all pipeline

    """

    problem_pipelines = []

    for p in ALL_PIPELINES:
        if p.problem_type == problem_type:
            problem_pipelines.append(p)

    if model_types is None:
        return problem_pipelines

    all_model_types = list_model_types()
    for model_type in model_types:
        if model_type not in all_model_types:
            raise RuntimeError("Unrecognized model type: %s" % model_type)

    pipelines = []

    for p in problem_pipelines:
        if p.model_type in model_types:
            pipelines.append(p)

    return pipelines


def list_model_types():
    return list(set([p.model_type for p in ALL_PIPELINES]))
