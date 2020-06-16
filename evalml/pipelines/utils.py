import numpy as np

from .binary_classification_pipeline import BinaryClassificationPipeline
from .classification import (
    CatBoostBinaryClassificationPipeline,
    CatBoostMulticlassClassificationPipeline,
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline,
    RFBinaryClassificationPipeline,
    RFMulticlassClassificationPipeline,
    XGBoostBinaryPipeline,
    XGBoostMulticlassPipeline
)
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression import (
    CatBoostRegressionPipeline,
    LinearRegressionPipeline,
    RFRegressionPipeline,
    XGBoostRegressionPipeline
)
from .regression_pipeline import RegressionPipeline

from evalml.exceptions import MissingComponentError
from evalml.model_family import handle_model_family
from evalml.pipelines.components import (
    DateTimeFeaturization,
    DropNullColumns,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler
)
from evalml.pipelines.components.estimators import Estimator
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger

logger = get_logger(__file__)

_ALL_PIPELINES = [CatBoostBinaryClassificationPipeline,
                  CatBoostMulticlassClassificationPipeline,
                  LogisticRegressionBinaryPipeline,
                  LogisticRegressionMulticlassPipeline,
                  RFBinaryClassificationPipeline,
                  RFMulticlassClassificationPipeline,
                  XGBoostBinaryPipeline,
                  XGBoostMulticlassPipeline,
                  CatBoostRegressionPipeline,
                  LinearRegressionPipeline,
                  RFRegressionPipeline,
                  XGBoostRegressionPipeline]


def all_pipelines():
    """Returns a complete list of all supported pipeline classes.

    Returns:
        list[PipelineBase]: a list of pipeline classes
    """
    pipelines = []
    for pipeline_class in _ALL_PIPELINES:
        try:
            pipeline_class({})
            pipelines.append(pipeline_class)
        except (MissingComponentError, ImportError):
            pipeline_name = pipeline_class.name
            logger.debug('Pipeline {} failed import, withholding from all_pipelines'.format(pipeline_name))
    return pipelines


def get_pipelines(problem_type, model_families=None):
    """Returns the pipelines allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:
        problem_type (ProblemTypes or str): problem type to filter for
        model_families (list[ModelFamily] or list[str]): model families to filter for

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


def all_estimators():
    """Returns a complete list of all supported estimator classes.

    Returns:
        list[Estimator]: a list of estimator classes
    """
    estimators = []
    for estimator_class in Estimator.__subclasses__():
        try:
            estimator_class()
            estimators.append(estimator_class)
        except (MissingComponentError, ImportError):
            estimator_name = estimator_class.name
            logger.debug('Estimator {} failed import, withholding from all_estimators'.format(estimator_name))
    return estimators


def get_estimators(problem_type, model_families=None):
    """Returns the estimators allowed for a particular problem type.

    Can also optionally filter by a list of model types.

    Arguments:
        problem_type (ProblemTypes or str): problem type to filter for
        model_families (list[ModelFamily] or list[str]): model families to filter for

    Returns:
        list[Estimator]: a list of estimator classes
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")

    if model_families:
        model_families = [handle_model_family(model_family) for model_family in model_families]

    problem_estimators = []
    problem_type = handle_problem_types(problem_type)
    for estimator in all_estimators():
        if problem_type in [handle_problem_types(supported_pt) for supported_pt in estimator.supported_problem_types]:
            problem_estimators.append(estimator)

    if model_families is None:
        return problem_estimators

    all_model_families = list_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    estimators = []

    for estimator in problem_estimators:
        if estimator.model_family in model_families:
            estimators.append(estimator)

    return estimators


def get_preprocessing_components(X, y, estimator):
    pp_components = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components += [DropNullColumns]
    X = X.drop(all_null_cols, axis=1)
    pp_components += [SimpleImputer]

    datetime_cols = X.select_dtypes(include=[np.datetime64])
    if len(datetime_cols.columns) > 0:
        pp_components += [DateTimeFeaturization]

    # DateTimeFeaturization can create categorical columns
    categorical_cols = X.select_dtypes(include=['category', 'object'])
    if len(datetime_cols.columns) > 0 or len(categorical_cols.columns) > 0:
        pp_components += [OneHotEncoder]

    if estimator is LinearRegressor or estimator is LogisticRegressionClassifier:
        pp_components += [StandardScaler]
    return pp_components


def make_pipeline(X, y, estimator, problem_type):
    preprocessing_components = get_preprocessing_components(X, y, estimator)
    complete_component_graph = preprocessing_components + [estimator]
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    if problem_type == ProblemTypes.BINARY:
        class GeneratedBinaryClassificationPipeline(BinaryClassificationPipeline):
            component_graph = complete_component_graph
        return GeneratedBinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        class GeneratedMulticlassClassificationPipeline(MulticlassClassificationPipeline):
            component_graph = complete_component_graph
        return GeneratedMulticlassClassificationPipeline
    elif problem_type == ProblemTypes.REGRESSION:
        class GeneratedRegressionPipeline(RegressionPipeline):
            component_graph = complete_component_graph
        return GeneratedRegressionPipeline
