from .random_forest import RFPipeline
from .xgboost import XGBoostPipeline

ALL_PIPELINES = [RFPipeline, XGBoostPipeline]


def get_pipelines(model_types=None):
    """Returns potential pipelines by model type

    Arguments:

        model_types(list[str]): model types to match. if none, return all pipelines

    Returns

        pipelines, list of all pipeline

    """

    if model_types is None:
        return ALL_PIPELINES

    pipelines = []

    for p in ALL_PIPELINES:
        if p.model_type in model_types:
            pipelines.append(p)

    return pipelines
