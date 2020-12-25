import json

from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline
)
from .regression_pipeline import RegressionPipeline
from .time_series_classification_pipelines import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline
)
from .time_series_regression_pipeline import TimeSeriesRegressionPipeline

from evalml.model_family import ModelFamily
from evalml.pipelines import PipelineBase
from evalml.pipelines.components import (  # noqa: F401
    CatBoostClassifier,
    CatBoostRegressor,
    ComponentBase,
    DateTimeFeaturizer,
    DropNullColumns,
    Estimator,
    Imputer,
    OneHotEncoder,
    RandomForestClassifier,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
    StandardScaler,
    TextFeaturizer
)
from evalml.pipelines.components.utils import all_components, get_estimators
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.utils import get_logger
from evalml.utils.gen_utils import _convert_to_woodwork_structure

logger = get_logger(__file__)


def _get_preprocessing_components(X, y, problem_type, text_columns, estimator_class):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Arguments:
        X (ww.DataTable): The input data of shape [n_samples, n_features]
        y (ww.DataColumn): The target data of length [n_samples]
        problem_type (ProblemTypes or str): Problem type
        text_columns (list): feature names which should be treated as text features
        estimator_class (class): A class which subclasses Estimator estimator for pipeline

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator
    """

    X_pd = X.to_dataframe()
    pp_components = []
    all_null_cols = X_pd.columns[X_pd.isnull().all()]
    if len(all_null_cols) > 0:
        pp_components.append(DropNullColumns)

    pp_components.append(Imputer)

    if text_columns:
        pp_components.append(TextFeaturizer)

    datetime_cols = X.select(["Datetime"])
    add_datetime_featurizer = len(datetime_cols.columns) > 0
    if add_datetime_featurizer:
        pp_components.append(DateTimeFeaturizer)

    # DateTimeFeaturizer can create categorical columns
    categorical_cols = X.select('category')
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
    elif problem_type == ProblemTypes.TIME_SERIES_REGRESSION:
        return TimeSeriesRegressionPipeline
    elif problem_type == ProblemTypes.TIME_SERIES_BINARY:
        return TimeSeriesBinaryClassificationPipeline
    else:
        return TimeSeriesMulticlassClassificationPipeline


def make_pipeline(X, y, estimator, problem_type, custom_hyperparameters=None, text_columns=None):
    """Given input data, target data, an estimator class and the problem type,
        generates a pipeline class with a preprocessing chain which was recommended based on the inputs.
        The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

   Arguments:
        X (pd.DataFrame, ww.DataTable): The input data of shape [n_samples, n_features]
        y (pd.Series, ww.DataColumn): The target data of length [n_samples]
        estimator (Estimator): Estimator for pipeline
        problem_type (ProblemTypes or str): Problem type for pipeline to generate
        custom_hyperparameters (dictionary): Dictionary of custom hyperparameters,
            with component name as key and dictionary of parameters as the value
        text_columns (list): feature names which should be treated as text features. Defaults to None.

    Returns:
        class: PipelineBase subclass with dynamically generated preprocessing components and specified estimator

    """
    X = _convert_to_woodwork_structure(X)
    y = _convert_to_woodwork_structure(y)

    problem_type = handle_problem_types(problem_type)
    if estimator not in get_estimators(problem_type):
        raise ValueError(f"{estimator.name} is not a valid estimator for problem type")
    preprocessing_components = _get_preprocessing_components(X, y, problem_type, text_columns, estimator)
    complete_component_graph = preprocessing_components + [estimator]

    if custom_hyperparameters and not isinstance(custom_hyperparameters, dict):
        raise ValueError(f"if custom_hyperparameters provided, must be dictionary. Received {type(custom_hyperparameters)}")

    hyperparameters = custom_hyperparameters
    base_class = _get_pipeline_base_class(problem_type)

    class GeneratedPipeline(base_class):
        custom_name = f"{estimator.name} w/ {' + '.join([component.name for component in preprocessing_components])}"
        component_graph = complete_component_graph
        custom_hyperparameters = hyperparameters

    return GeneratedPipeline


def make_pipeline_from_components(component_instances, problem_type, custom_name=None, random_state=0):
    """Given a list of component instances and the problem type, an pipeline instance is generated with the component instances.
    The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type. The pipeline will be
    untrained, even if the input components are already trained. A custom name for the pipeline can optionally be specified;
    otherwise the default pipeline name will be 'Templated Pipeline'.

   Arguments:
        component_instances (list): a list of all of the components to include in the pipeline
        problem_type (str or ProblemTypes): problem type for the pipeline to generate
        custom_name (string): a name for the new pipeline
        random_state (int or np.random.RandomState): Random state used to intialize the pipeline.

    Returns:
        Pipeline instance with component instances and specified estimator created from given random state.

    Example:
        >>> components = [Imputer(), StandardScaler(), RandomForestClassifier()]
        >>> pipeline = make_pipeline_from_components(components, problem_type="binary")
        >>> pipeline.describe()

    """
    for i, component in enumerate(component_instances):
        if not isinstance(component, ComponentBase):
            raise TypeError("Every element of `component_instances` must be an instance of ComponentBase")
        if i == len(component_instances) - 1 and not isinstance(component, Estimator):
            raise ValueError("Pipeline needs to have an estimator at the last position of the component list")

    if custom_name and not isinstance(custom_name, str):
        raise TypeError("Custom pipeline name must be a string")
    pipeline_name = custom_name
    problem_type = handle_problem_types(problem_type)

    class TemplatedPipeline(_get_pipeline_base_class(problem_type)):
        custom_name = pipeline_name
        component_graph = [c.__class__ for c in component_instances]
    return TemplatedPipeline({c.name: c.parameters for c in component_instances}, random_state=random_state)


def generate_pipeline_code(element):
    """Creates and returns a string that contains the Python imports and code required for running the EvalML pipeline.

    Arguments:
        element (pipeline instance): The instance of the pipeline to generate string Python code

    Returns:
        String representation of Python code that can be run separately in order to recreate the pipeline instance.
        Does not include code for custom component implementation.
    """
    # hold the imports needed and add code to end
    code_strings = ['import json']
    if not isinstance(element, PipelineBase):
        raise ValueError("Element must be a pipeline instance, received {}".format(type(element)))
    if isinstance(element.component_graph, dict):
        raise ValueError("Code generation for nonlinear pipelines is not supported yet")

    component_graph_string = ',\n\t\t'.join([com.__class__.__name__ if com.__class__ not in all_components() else "'{}'".format(com.name) for com in element._component_graph])
    code_strings.append("from {} import {}".format(element.__class__.__bases__[0].__module__, element.__class__.__bases__[0].__name__))
    # check for other attributes associated with pipeline (ie name, custom_hyperparameters)
    pipeline_list = []
    for k, v in sorted(list(filter(lambda item: item[0][0] != '_', element.__class__.__dict__.items())), key=lambda x: x[0]):
        if k == 'component_graph':
            continue
        pipeline_list += ["{} = '{}'".format(k, v)] if isinstance(v, str) else ["{} = {}".format(k, v)]

    pipeline_string = "\t" + "\n\t".join(pipeline_list) + "\n" if len(pipeline_list) else ""

    try:
        ret = json.dumps(element.parameters, indent='\t')
    except TypeError:
        raise TypeError(f"Value {element.parameters} cannot be JSON-serialized")
    # create the base string for the pipeline
    base_string = "\nclass {0}({1}):\n" \
                  "\tcomponent_graph = [\n\t\t{2}\n\t]\n" \
                  "{3}" \
                  "\nparameters = json.loads(\"\"\"{4}\"\"\")\n" \
                  "pipeline = {0}(parameters)" \
                  .format(element.__class__.__name__,
                          element.__class__.__bases__[0].__name__,
                          component_graph_string,
                          pipeline_string,
                          ret)
    code_strings.append(base_string)
    return "\n".join(code_strings)


def _make_stacked_ensemble_pipeline(input_pipelines, problem_type, random_state=0):
    """
    Creates a pipeline with a stacked ensemble estimator.

    Arguments:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators for the stacked ensemble.
            This must not be None or an empty list or else EnsembleMissingPipelinesError will be raised.
        problem_type (ProblemType): problem type of pipeline

    Returns:
        Pipeline with appropriate stacked ensemble estimator.
    """
    if problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
        return make_pipeline_from_components([StackedEnsembleClassifier(input_pipelines)], problem_type,
                                             custom_name="Stacked Ensemble Classification Pipeline",
                                             random_state=random_state)
    else:
        return make_pipeline_from_components([StackedEnsembleRegressor(input_pipelines)], problem_type,
                                             custom_name="Stacked Ensemble Regression Pipeline",
                                             random_state=random_state)
