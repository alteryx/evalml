from .logistic_regression import LogisticRegressionPipeline
from .random_forest import RFPipeline
from .xgboost import XGBoostPipeline

ALL_PIPELINES = [RFPipeline, XGBoostPipeline, LogisticRegressionPipeline]


def get_pipelines(model_types=None):
    """Returns potential pipelines by model type

    Arguments:

        model_types(list[str]): model types to match. if none, return all pipelines

    Returns

        pipelines, list of all pipeline

    """

    if model_types is None:
        return ALL_PIPELINES

    all_model_types = list_model_types()
    for model_type in model_types:
        if model_type not in all_model_types:
            raise RuntimeError("Unrecognized model type: %s" % model_type)

    pipelines = []

    for p in ALL_PIPELINES:
        if p.model_type in model_types:
            pipelines.append(p)

    return pipelines


def list_model_types():
    return list(set([p.model_type for p in ALL_PIPELINES]))
