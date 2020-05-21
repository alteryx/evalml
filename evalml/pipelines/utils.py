import copy

from .classification import (
    CatBoostBinaryClassificationPipeline,
    CatBoostMulticlassClassificationPipeline,
    ETBinaryClassificationPipeline,
    ETMulticlassClassificationPipeline,
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline,
    RFBinaryClassificationPipeline,
    RFMulticlassClassificationPipeline,
    XGBoostBinaryPipeline,
    XGBoostMulticlassPipeline
)
from .regression import (
    CatBoostRegressionPipeline,
    ETRegressionPipeline,
    LinearRegressionPipeline,
    RFRegressionPipeline,
    XGBoostRegressionPipeline
)

from evalml.model_family import handle_model_family
from evalml.problem_types import handle_problem_types
from evalml.utils import import_or_raise

_ALL_PIPELINES = [CatBoostBinaryClassificationPipeline,
                  CatBoostMulticlassClassificationPipeline,
                  ETBinaryClassificationPipeline,
                  ETMulticlassClassificationPipeline,
                  LogisticRegressionBinaryPipeline,
                  LogisticRegressionMulticlassPipeline,
                  RFBinaryClassificationPipeline,
                  RFMulticlassClassificationPipeline,
                  XGBoostBinaryPipeline,
                  XGBoostMulticlassPipeline,
                  LinearRegressionPipeline,
                  RFRegressionPipeline,
                  CatBoostRegressionPipeline,
                  XGBoostRegressionPipeline,
                  ETRegressionPipeline]


def all_pipelines():
    """Returns a complete list of all supported pipeline classes.

    Returns:
        list[PipelineBase]: a list of pipeline classes
    """
    pipelines = copy.copy(_ALL_PIPELINES)
    try:
        import_or_raise("xgboost", error_msg="XGBoost not installed.")
    except ImportError:
        pipelines.remove(XGBoostBinaryPipeline)
        pipelines.remove(XGBoostMulticlassPipeline)
        pipelines.remove(XGBoostRegressionPipeline)
    try:
        import_or_raise("catboost", error_msg="Catboost not installed.")
    except ImportError:
        pipelines.remove(CatBoostBinaryClassificationPipeline)
        pipelines.remove(CatBoostMulticlassClassificationPipeline)
        pipelines.remove(CatBoostRegressionPipeline)
    return pipelines


def get_pipelines(problem_type, model_families=None):
    """Returns the pipelines allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:

    Returns:
        list[PipelineBase]: a list of pipeline classes
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")

    if model_families:
        model_families = [handle_model_family(model_family) for model_family in model_families]

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in all_pipelines():
        if problem_type == handle_problem_types(p.problem_type):
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
        list[ModelFamily]: a list of model families
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in all_pipelines():
        if problem_type == handle_problem_types(p.problem_type):
            problem_pipelines.append(p)

    return list(set([p.model_family for p in problem_pipelines]))
