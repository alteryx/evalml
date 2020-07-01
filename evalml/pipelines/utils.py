import numpy as np
import pandas as pd

from .binary_classification_pipeline import BinaryClassificationPipeline
from .classification import (
    LogisticRegressionBinaryPipeline,
    LogisticRegressionMulticlassPipeline
)
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression import LinearRegressionPipeline
from .regression_pipeline import RegressionPipeline

from evalml.exceptions import MissingComponentError
from evalml.model_family import handle_model_family
from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturization,
    DropNullColumns,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    RandomForestClassifier,
    RandomForestRegressor,
    SimpleImputer,
    StandardScaler,
    XGBoostClassifier,
    XGBoostRegressor
)
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger

logger = get_logger(__file__)

_ALL_PIPELINES = [LogisticRegressionBinaryPipeline,
                  LogisticRegressionMulticlassPipeline,
                  LinearRegressionPipeline]


_ALL_ESTIMATORS = [CatBoostClassifier,
                   CatBoostRegressor,
                   LinearRegressor,
                   LogisticRegressionClassifier,
                   RandomForestClassifier,
                   RandomForestRegressor,
                   XGBoostClassifier,
                   XGBoostRegressor]


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
    """List model type for a particular problem type.

    Arguments:
        problem_types (ProblemTypes or str): binary, multiclass, or regression

    Returns:
        list[ModelFamily]: a list of model families
    """

    problem_pipelines = []
    problem_type = handle_problem_types(problem_type)
    for p in all_estimators():
        if problem_type == handle_problem_types(p.problem_type):
            problem_pipelines.append(p)

    return list(set([p.model_family for p in problem_pipelines]))


def all_estimators():
    """Returns a complete list of all supported estimator classes.

    Returns:
        list[Estimator]: a list of estimator classes
    """
    estimators = []
    for estimator_class in _ALL_ESTIMATORS:
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
        list[class]: a list of estimator subclasses
    """
    if model_families is not None and not isinstance(model_families, list):
        raise TypeError("model_families parameter is not a list.")
    problem_type = handle_problem_types(problem_type)
    if model_families is None:
        model_families = list_model_families(problem_type)

    model_families = [handle_model_family(model_family) for model_family in model_families]
    all_model_families = list_model_families(problem_type)
    for model_family in model_families:
        if model_family not in all_model_families:
            raise RuntimeError("Unrecognized model type for problem type %s: %s" % (problem_type, model_family))

    estimator_classes = []
    for estimator_class in all_estimators():
        if problem_type not in [handle_problem_types(supported_pt) for supported_pt in estimator_class.supported_problem_types]:
            continue
        if estimator_class.model_family not in model_families:
            continue
        estimator_classes.append(estimator_class)
    return estimator_classes


def _get_preprocessing_components(X, y, problem_type, estimator_class):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (pd.DataFrame): the input data of shape [n_samples, n_features]
        y (pd.Series): the target labels of length [n_samples]
        problem_type (ProblemTypes or str): problem type
        estimator_class (class):A class which subclasses Estimator estimator for pipeline

    Returns:
        list[Transformer]: a list of applicable preprocessing components to use with the estimator
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pp_components = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)
    pp_components.append(SimpleImputer)

    datetime_cols = X.select_dtypes(include=[np.datetime64])
    add_datetime_featurization = len(datetime_cols.columns) > 0
    if add_datetime_featurization:
        pp_components.append(DateTimeFeaturization)

    # DateTimeFeaturization can create categorical columns
    categorical_cols = X.select_dtypes(include=['category', 'object'])
    if (add_datetime_featurization or len(categorical_cols.columns) > 0) and estimator_class not in {CatBoostClassifier, CatBoostRegressor}:
        pp_components.append(OneHotEncoder)

    if estimator_class in {LinearRegressor, LogisticRegressionClassifier}:
        pp_components.append(StandardScaler)
    return pp_components


def make_pipeline(X, y, estimator, problem_type):
    """Given input data, target data, an estimator class and the problem type,
        generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
        The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

   Arguments:
        X (pd.DataFrame): the input data of shape [n_samples, n_features]
        y (pd.Series): the target labels of length [n_samples]
        estimator (Estimator): estimator for pipeline
        problem_type (ProblemTypes or str): problem type for pipeline to generate

    Returns:
        class: PipelineBase subclass with dynamically generated preprocessing components and specified estimator

    """
    problem_type = handle_problem_types(problem_type)
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    preprocessing_components = _get_preprocessing_components(X, y, problem_type, estimator)
    complete_component_graph = preprocessing_components + [estimator]

    hyperparameters = None
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    categorical_cols = X.select_dtypes(include=['category', 'object'])
    if estimator in {CatBoostClassifier, CatBoostRegressor} or len(categorical_cols.columns) > 0:
        # a workaround to avoid choosing an impute_strategy which won't work with categorical inputs
        logger.debug("Limiting SimpleImputer to use 'most_frequent' strategy to avoid choosing an impute strategy that won't work with categorical inputs.")
        hyperparameters = {
            'Simple Imputer': {
                "impute_strategy": ["most_frequent"]
            }
        }

    def get_pipeline_base_class(problem_type):
        """Returns pipeline base class for problem_type"""
        if problem_type == ProblemTypes.BINARY:
            return BinaryClassificationPipeline
        elif problem_type == ProblemTypes.MULTICLASS:
            return MulticlassClassificationPipeline
        elif problem_type == ProblemTypes.REGRESSION:
            return RegressionPipeline

    base_class = get_pipeline_base_class(problem_type)

    class GeneratedPipeline(base_class):
        custom_name = f"{estimator.name} w/ {' + '.join([component.name for component in preprocessing_components])}"
        component_graph = complete_component_graph
        custom_hyperparameters = hyperparameters

    return GeneratedPipeline
