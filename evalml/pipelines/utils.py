import numpy as np
import pandas as pd

from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression_pipeline import RegressionPipeline

from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturizer,
    DropNullColumns,
    Imputer,
    LinearRegressor,
    LogisticRegressionClassifier,
    OneHotEncoder,
    StandardScaler
)
from evalml.pipelines.components.utils import get_estimators
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger

logger = get_logger(__file__)


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

    pp_components.append(Imputer)

    datetime_cols = X.select_dtypes(include=[np.datetime64])
    add_datetime_featurizer = len(datetime_cols.columns) > 0
    if add_datetime_featurizer:
        pp_components.append(DateTimeFeaturizer)

    # DateTimeFeaturizer can create categorical columns
    categorical_cols = X.select_dtypes(include=['category', 'object'])
    if (add_datetime_featurizer or len(categorical_cols.columns) > 0) and estimator_class not in {CatBoostClassifier, CatBoostRegressor}:
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
