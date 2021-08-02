from woodwork import logical_types

from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from .pipeline_base import PipelineBase
from .regression_pipeline import RegressionPipeline
from .time_series_classification_pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
)
from .time_series_regression_pipeline import TimeSeriesRegressionPipeline

from evalml.data_checks import DataCheckActionCode, TargetDistributionDataCheck
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (  # noqa: F401
    CatBoostClassifier,
    CatBoostRegressor,
    ComponentBase,
    DateTimeFeaturizer,
    DelayedFeatureTransformer,
    DropColumns,
    DropNullColumns,
    EmailFeaturizer,
    Estimator,
    Imputer,
    LogTransformer,
    OneHotEncoder,
    RandomForestClassifier,
    SMOTENCSampler,
    SMOTENSampler,
    SMOTESampler,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
    StandardScaler,
    TargetImputer,
    TextFeaturizer,
    Undersampler,
    URLFeaturizer,
)
from evalml.pipelines.components.utils import get_estimators
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_classification,
    is_regression,
    is_time_series,
)
from evalml.utils import get_logger, import_or_raise, infer_feature_types

logger = get_logger(__file__)


def _get_preprocessing_components(
    X, y, problem_type, estimator_class, sampler_name=None
):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (pd.DataFrame): The input data of shape [n_samples, n_features]
        y (pd.Series): The target data of length [n_samples]
        problem_type (ProblemTypes or str): Problem type
        estimator_class (class): A class which subclasses Estimator estimator for pipeline,
        sampler_name (str): The name of the sampler component to add to the pipeline. Defaults to None

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator
    """

    pp_components = []

    if is_regression(problem_type):
        for each_action in TargetDistributionDataCheck().validate(X, y)["actions"]:
            if each_action["metadata"]["transformation_strategy"] == "lognormal":
                pp_components.append(LogTransformer)

    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)

    email_columns = list(X.ww.select("EmailAddress", return_schema=True).columns)
    if len(email_columns) > 0:
        pp_components.append(EmailFeaturizer)

    url_columns = list(X.ww.select("URL", return_schema=True).columns)
    if len(url_columns) > 0:
        pp_components.append(URLFeaturizer)

    input_logical_types = {type(lt) for lt in X.ww.logical_types.values()}
    types_imputer_handles = {
        logical_types.Boolean,
        logical_types.Categorical,
        logical_types.Double,
        logical_types.Integer,
        logical_types.URL,
        logical_types.EmailAddress,
    }

    text_columns = list(X.ww.select("NaturalLanguage", return_schema=True).columns)
    if len(text_columns) > 0:
        pp_components.append(TextFeaturizer)

    if len(input_logical_types.intersection(types_imputer_handles)) or len(
        text_columns
    ):
        pp_components.append(Imputer)

    index_and_unknown_columns = list(
        X.ww.select(["index", "unknown"], return_schema=True).columns
    )
    if len(index_and_unknown_columns) > 0:
        pp_components.append(DropColumns)

    datetime_cols = list(X.ww.select(["Datetime"], return_schema=True).columns)

    add_datetime_featurizer = len(datetime_cols) > 0
    if add_datetime_featurizer and estimator_class.model_family != ModelFamily.ARIMA:
        pp_components.append(DateTimeFeaturizer)

    if (
        is_time_series(problem_type)
        and estimator_class.model_family != ModelFamily.ARIMA
    ):
        pp_components.append(DelayedFeatureTransformer)

    # The URL and EmailAddress Featurizers will create categorical columns
    categorical_cols = list(
        X.ww.select(["category", "URL", "EmailAddress"], return_schema=True).columns
    )
    if len(categorical_cols) > 0 and estimator_class not in {
        CatBoostClassifier,
        CatBoostRegressor,
    }:
        pp_components.append(OneHotEncoder)

    sampler_components = {
        "Undersampler": Undersampler,
        "SMOTE Oversampler": SMOTESampler,
        "SMOTENC Oversampler": SMOTENCSampler,
        "SMOTEN Oversampler": SMOTENSampler,
    }
    if sampler_name is not None:
        try:
            import_or_raise(
                "imblearn.over_sampling", error_msg="imbalanced-learn is not installed"
            )
            pp_components.append(sampler_components[sampler_name])
        except ImportError:
            logger.debug(
                f"Could not import imblearn.over_sampling, so defaulting to use Undersampler"
            )
            pp_components.append(Undersampler)

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
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        return TimeSeriesRegressionPipeline
    elif problem_type == ProblemTypes.TIME_SERIES_BINARY:
        return TimeSeriesBinaryClassificationPipeline
    else:
        return TimeSeriesMulticlassClassificationPipeline


def make_pipeline(
    X,
    y,
    estimator,
    problem_type,
    parameters=None,
    sampler_name=None,
    extra_components=[],
):
    """Given input data, target data, an estimator class and the problem type,
         generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
         The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

    Arguments:
         X (pd.DataFrame): The input data of shape [n_samples, n_features]
         y (pd.Series): The target data of length [n_samples]
         estimator (Estimator): Estimator for pipeline
         problem_type (ProblemTypes or str): Problem type for pipeline to generate
         parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters.
         sampler_name (str): The name of the sampler component to add to the pipeline. Only used in classification problems.
             Defaults to None
         extra_components (list(ComponentBase)): list of extra components to be added after preprocessing components.

     Returns:
         PipelineBase object: PipelineBase instance with dynamically generated preprocessing components and specified estimator

    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    problem_type = handle_problem_types(problem_type)
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    if not is_classification(problem_type) and sampler_name is not None:
        raise ValueError(
            f"Sampling is unsupported for problem_type {str(problem_type)}"
        )
    preprocessing_components = _get_preprocessing_components(
        X, y, problem_type, estimator, sampler_name
    )

    complete_component_list = preprocessing_components + extra_components + [estimator]
    component_graph = PipelineBase._make_component_dict_from_component_list(
        complete_component_list
    )
    base_class = _get_pipeline_base_class(problem_type)
    return base_class(
        component_graph,
        parameters=parameters,
    )


def generate_pipeline_code(element):
    """Creates and returns a string that contains the Python imports and code required for running the EvalML pipeline.

    Arguments:
        element (pipeline instance): The instance of the pipeline to generate string Python code

    Returns:
        String representation of Python code that can be run separately in order to recreate the pipeline instance.
        Does not include code for custom component implementation.
    """
    # hold the imports needed and add code to end
    code_strings = []
    if not isinstance(element, PipelineBase):
        raise ValueError(
            "Element must be a pipeline instance, received {}".format(type(element))
        )
    if isinstance(element.component_graph, dict):
        raise ValueError("Code generation for nonlinear pipelines is not supported yet")
    code_strings.append(
        "from {} import {}".format(
            element.__class__.__module__, element.__class__.__name__
        )
    )
    code_strings.append(repr(element))
    return "\n".join(code_strings)


def _make_stacked_ensemble_pipeline(
    input_pipelines, problem_type, n_jobs=-1, random_seed=0
):
    """
    Creates a pipeline with a stacked ensemble estimator.

    Arguments:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators for the stacked ensemble.
            This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
        problem_type (ProblemType): problem type of pipeline
        n_jobs (int or None): Integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            Defaults to -1.

    Returns:
        Pipeline with appropriate stacked ensemble estimator.
    """
    parameters = {}
    if is_classification(problem_type):
        parameters = {
            "Stacked Ensemble Classifier": {
                "input_pipelines": input_pipelines,
                "n_jobs": n_jobs,
            }
        }
        estimator = StackedEnsembleClassifier
    else:
        parameters = {
            "Stacked Ensemble Regressor": {
                "input_pipelines": input_pipelines,
                "n_jobs": n_jobs,
            }
        }
        estimator = StackedEnsembleRegressor

    pipeline_class, pipeline_name = {
        ProblemTypes.BINARY: (
            BinaryClassificationPipeline,
            "Stacked Ensemble Classification Pipeline",
        ),
        ProblemTypes.MULTICLASS: (
            MulticlassClassificationPipeline,
            "Stacked Ensemble Classification Pipeline",
        ),
        ProblemTypes.REGRESSION: (
            RegressionPipeline,
            "Stacked Ensemble Regression Pipeline",
        ),
    }[problem_type]

    return pipeline_class(
        [estimator],
        parameters=parameters,
        custom_name=pipeline_name,
        random_seed=random_seed,
    )


def _make_component_list_from_actions(actions):
    """
    Creates a list of components from the input DataCheckAction list

    Arguments:
        actions (list(DataCheckAction)): List of DataCheckAction objects used to create list of components

    Returns:
        List of components used to address the input actions
    """
    components = []
    for action in actions:
        if action.action_code == DataCheckActionCode.DROP_COL:
            components.append(DropColumns(columns=action.metadata["columns"]))
        if action.action_code == DataCheckActionCode.IMPUTE_COL:
            metadata = action.metadata
            if metadata["is_target"]:
                components.append(
                    TargetImputer(impute_strategy=metadata["impute_strategy"])
                )
    return components
