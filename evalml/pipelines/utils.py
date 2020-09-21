import pandas as pd

from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression_pipeline import RegressionPipeline

from evalml.model_family import ModelFamily
from evalml.pipelines.components import (
    CatBoostClassifier,
    CatBoostRegressor,
    DateTimeFeaturizer,
    DropNullColumns,
    Estimator,
    Imputer,
    OneHotEncoder,
    StandardScaler
)
from evalml.pipelines.components.utils import get_estimators
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger
from evalml.utils.gen_utils import categorical_dtypes, datetime_dtypes

logger = get_logger(__file__)


def _get_preprocessing_components(X, y, problem_type, estimator_class):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (pd.DataFrame): The input data of shape [n_samples, n_features]
        y (pd.Series): The target data of length [n_samples]
        problem_type (ProblemTypes or str): Problem type
        estimator_class (class): A class which subclasses Estimator estimator for pipeline

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    pp_components = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)

    pp_components.append(Imputer)

    datetime_cols = X.select_dtypes(include=datetime_dtypes)
    add_datetime_featurizer = len(datetime_cols.columns) > 0
    if add_datetime_featurizer:
        pp_components.append(DateTimeFeaturizer)

    # DateTimeFeaturizer can create categorical columns
    categorical_cols = X.select_dtypes(include=categorical_dtypes)
    if (add_datetime_featurizer or len(categorical_cols.columns) > 0) and estimator_class not in {CatBoostClassifier, CatBoostRegressor}:
        pp_components.append(OneHotEncoder)

    if estimator_class.model_family == ModelFamily.LINEAR_MODEL:
        pp_components.append(StandardScaler)
    return pp_components


def _get_pipeline_base_class(problem_type):
    """Returns pipeline base class for problem_type"""
    if problem_type == ProblemTypes.BINARY:
        return BinaryClassificationPipeline
    elif problem_type == ProblemTypes.MULTICLASS:
        return MulticlassClassificationPipeline
    elif problem_type == ProblemTypes.REGRESSION:
        return RegressionPipeline


def make_pipeline(X, y, estimator, problem_type):
    """Given input data, target data, an estimator class and the problem type,
        generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
        The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

   Arguments:
        X (pd.DataFrame): The input data of shape [n_samples, n_features]
        y (pd.Series): The target data of length [n_samples]
        estimator (Estimator): Estimator for pipeline
        problem_type (ProblemTypes or str): Problem type for pipeline to generate

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

    base_class = _get_pipeline_base_class(problem_type)

    class GeneratedPipeline(base_class):
        custom_name = f"{estimator.name} w/ {' + '.join([component.name for component in preprocessing_components])}"
        component_graph = complete_component_graph
        custom_hyperparameters = hyperparameters

    return GeneratedPipeline


def make_pipeline_from_components(component_instances, problem_type, custom_name=None):
    """Given a list of component instances and the problem type, a pipeline instance is generated with the component instances.
    The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type. A custom name for
    the pipeline can optionally be specified; otherwise the default pipeline name will be 'Templated Pipeline'.

   Arguments:
        component_instances (list): a list of all of the components to include in the pipeline
        problem_type (str or ProblemTypes): problem type for the pipeline to generate
        custom_name (string): a name for the new pipeline

    Returns:
        Pipeline instance with component instances and specified estimator

    Example:
        >>> components = [Imputer(), StandardScaler(), CatBoostClassifier()]
        >>> pipeline = make_pipeline_from_components(components, problem_type="binary")
        >>> pipeline.describe()
        >>> assert pipeline.components_graph == components

    """
    if not isinstance(component_instances[-1], Estimator):
        raise ValueError("Pipeline needs to have an estimator at the last position of the component list")

    pipeline_name = custom_name
    problem_type = handle_problem_types(problem_type)

    class TemplatedPipeline(_get_pipeline_base_class(problem_type)):
        custom_name = pipeline_name
        component_graph = [c.__class__ for c in component_instances]
    return TemplatedPipeline({c.name: c.parameters for c in component_instances})