import cloudpickle

from .classification import (
    LogisticRegressionPipeline,
    RFClassificationPipeline,
    XGBoostPipeline
)
from .regression import LinearRegressionPipeline, RFRegressionPipeline

from evalml.pipelines.components import (
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RFClassifierSelectFromModel,
    SimpleImputer,
    StandardScaler,
    XGBoostClassifier,
    LinearRegressor,
    RandomForestRegressor,
    RFRegressorSelectFromModel,
)
from evalml.model_types import handle_model_types
from evalml.problem_types import handle_problem_types
from evalml.automl.pipeline_template import PipelineTemplate
# ALL_PIPELINES = [RFClassificationPipeline, XGBoostPipeline, LogisticRegressionPipeline, LinearRegressionPipeline, RFRegressionPipeline]

def get_classification_templates():
    rfc = [OneHotEncoder, SimpleImputer, RFClassifierSelectFromModel, RandomForestClassifier]
    xgb = [OneHotEncoder, SimpleImputer, RFClassifierSelectFromModel, XGBoostClassifier]
    lgr = [OneHotEncoder, SimpleImputer, StandardScaler, LogisticRegressionClassifier]
    pipelines = [rfc, xgb, lgr]
    templates = []
    for pipeline in pipelines:
        template = PipelineTemplate(pipeline)
        templates.append(template)
    return templates

def get_regression_templates():
    rfr = [OneHotEncoder, SimpleImputer, RFRegressorSelectFromModel, RandomForestRegressor]
    lrp = [OneHotEncoder, SimpleImputer, StandardScaler, LinearRegressor]
    pipelines = [rfr, lrp]
    templates = []
    for pipeline in pipelines:
        template = PipelineTemplate(pipeline)
        templates.append(template)
    return templates


def get_all_templates():
    classification = get_classification_templates()
    regression = get_regression_templates()
    return classification + regression


def list_model_types(problem_type):
    """List model type for a particular problem type

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        model_types, list of model types
    """

    problem_templates = []
    problem_type = handle_problem_types(problem_type)
    templates = get_all_templates()
    for t in templates:
        if problem_type in t.problem_types:
            problem_pipelines.append(t)

    return list(set([t.model_type for t in problem_templates]))


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
    templates = get_all_templates()
    for p in templates:
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
