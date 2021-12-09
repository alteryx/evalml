"""Utility methods for EvalML pipelines."""
import logging

from woodwork import logical_types

from . import (
    TimeSeriesBinaryClassificationPipeline,
    TimeSeriesMulticlassClassificationPipeline,
    TimeSeriesRegressionPipeline,
)
from .binary_classification_pipeline import BinaryClassificationPipeline
from .multiclass_classification_pipeline import (
    MulticlassClassificationPipeline,
)
from .pipeline_base import PipelineBase
from .regression_pipeline import RegressionPipeline

from evalml.data_checks import DataCheckActionCode
from evalml.model_family import ModelFamily
from evalml.pipelines.components import (  # noqa: F401
    CatBoostClassifier,
    CatBoostRegressor,
    ComponentBase,
    DateTimeFeaturizer,
    DropColumns,
    DropNullColumns,
    DropRowsTransformer,
    EmailFeaturizer,
    Estimator,
    Imputer,
    LogTransformer,
    NaturalLanguageFeaturizer,
    OneHotEncoder,
    Oversampler,
    RandomForestClassifier,
    StackedEnsembleClassifier,
    StackedEnsembleRegressor,
    StandardScaler,
    TargetImputer,
    TimeSeriesFeaturizer,
    Undersampler,
    URLFeaturizer,
)
from evalml.pipelines.components.transformers.encoders.label_encoder import (
    LabelEncoder,
)
from evalml.pipelines.components.utils import (
    get_estimators,
    handle_component_class,
)
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_classification,
    is_time_series,
)
from evalml.utils import import_or_raise, infer_feature_types

logger = logging.getLogger(__name__)


def _get_label_encoder(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    if is_classification(problem_type):
        component.append(LabelEncoder)
    return component


def _get_drop_all_null(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    all_null_cols = X.columns[X.isnull().all()]
    if len(all_null_cols) > 0:
        component.append(DropNullColumns)
    return component


def _get_drop_index_unknown(X, y, problem_type, estimator_class, sampler_name=None):
    component = []
    index_and_unknown_columns = list(
        X.ww.select(["index", "unknown"], return_schema=True).columns
    )
    if len(index_and_unknown_columns) > 0:
        component.append(DropColumns)
    return component


def _get_url_email(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    email_columns = list(X.ww.select("EmailAddress", return_schema=True).columns)
    if len(email_columns) > 0:
        components.append(EmailFeaturizer)

    url_columns = list(X.ww.select("URL", return_schema=True).columns)
    if len(url_columns) > 0:
        components.append(URLFeaturizer)

    return components


def _get_datetime(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    datetime_cols = list(X.ww.select(["Datetime"], return_schema=True).columns)

    add_datetime_featurizer = len(datetime_cols) > 0
    if add_datetime_featurizer and estimator_class.model_family not in [
        ModelFamily.ARIMA,
        ModelFamily.PROPHET,
    ]:
        components.append(DateTimeFeaturizer)
    return components


def _get_natural_language(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    text_columns = list(X.ww.select("NaturalLanguage", return_schema=True).columns)
    if len(text_columns) > 0:
        components.append(NaturalLanguageFeaturizer)
    return components


def _get_imputer(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    input_logical_types = {type(lt) for lt in X.ww.logical_types.values()}
    text_columns = list(X.ww.select("NaturalLanguage", return_schema=True).columns)

    types_imputer_handles = {
        logical_types.Boolean,
        logical_types.Categorical,
        logical_types.Double,
        logical_types.Integer,
        logical_types.URL,
        logical_types.EmailAddress,
        logical_types.Datetime,
    }

    if len(input_logical_types.intersection(types_imputer_handles)) or len(
        text_columns
    ):
        components.append(Imputer)

    return components


def _get_ohe(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    # The URL and EmailAddress Featurizers will create categorical columns
    categorical_cols = list(
        X.ww.select(["category", "URL", "EmailAddress"], return_schema=True).columns
    )
    if len(categorical_cols) > 0 and estimator_class not in {
        CatBoostClassifier,
        CatBoostRegressor,
    }:
        components.append(OneHotEncoder)
    return components


def _get_sampler(X, y, problem_type, estimator_class, sampler_name=None):
    components = []

    sampler_components = {
        "Undersampler": Undersampler,
        "Oversampler": Oversampler,
    }
    if sampler_name is not None:
        try:
            import_or_raise(
                "imblearn.over_sampling", error_msg="imbalanced-learn is not installed"
            )
            components.append(sampler_components[sampler_name])
        except ImportError:
            logger.warning(
                "Could not import imblearn.over_sampling, so defaulting to use Undersampler"
            )
            components.append(Undersampler)
    return components


def _get_standard_scaler(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    if estimator_class and estimator_class.model_family == ModelFamily.LINEAR_MODEL:
        components.append(StandardScaler)
    return components


def _get_time_series_featurizer(X, y, problem_type, estimator_class, sampler_name=None):
    components = []
    if (
        is_time_series(problem_type)
        and estimator_class.model_family != ModelFamily.ARIMA
    ):
        components.append(TimeSeriesFeaturizer)
    return components


def _get_preprocessing_components(
    X, y, problem_type, estimator_class, sampler_name=None
):
    """Given input data, target data and an estimator class, construct a recommended preprocessing chain to be combined with the estimator and trained on the provided data.

    Args:
        X (pd.DataFrame): The input data of shape [n_samples, n_features].
        y (pd.Series): The target data of length [n_samples].
        problem_type (ProblemTypes or str): Problem type.
        estimator_class (class): A class which subclasses Estimator estimator for pipeline.
        sampler_name (str): The name of the sampler component to add to the pipeline. Defaults to None.

    Returns:
        list[Transformer]: A list of applicable preprocessing components to use with the estimator.
    """
    if is_time_series(problem_type):
        components_functions = [
            _get_label_encoder,
            _get_drop_all_null,
            _get_drop_index_unknown,
            _get_url_email,
            _get_natural_language,
            _get_imputer,
            _get_time_series_featurizer,
            _get_datetime,
            _get_ohe,
            _get_sampler,
            _get_standard_scaler,
        ]
    else:
        components_functions = [
            _get_label_encoder,
            _get_drop_all_null,
            _get_drop_index_unknown,
            _get_url_email,
            _get_datetime,
            _get_natural_language,
            _get_imputer,
            _get_ohe,
            _get_sampler,
            _get_standard_scaler,
        ]
    components = []
    for function in components_functions:
        components.extend(function(X, y, problem_type, estimator_class, sampler_name))

    return components


def _get_pipeline_base_class(problem_type):
    """Returns pipeline base class for problem_type."""
    problem_type = handle_problem_types(problem_type)
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
    extra_components=None,
    extra_components_position="before_preprocessing",
    use_estimator=True,
):
    """Given input data, target data, an estimator class and the problem type, generates a pipeline class with a preprocessing chain which was recommended based on the inputs. The pipeline will be a subclass of the appropriate pipeline base class for the specified problem_type.

    Args:
         X (pd.DataFrame): The input data of shape [n_samples, n_features].
         y (pd.Series): The target data of length [n_samples].
         estimator (Estimator): Estimator for pipeline.
         problem_type (ProblemTypes or str): Problem type for pipeline to generate.
         parameters (dict): Dictionary with component names as keys and dictionary of that component's parameters as values.
             An empty dictionary or None implies using all default values for component parameters.
         sampler_name (str): The name of the sampler component to add to the pipeline. Only used in classification problems.
             Defaults to None
         extra_components (list[ComponentBase]): List of extra components to be added after preprocessing components. Defaults to None.
         extra_components_position (str): Where to put extra components. Defaults to "before_preprocessing" and any other value will put components after preprocessing components.
         use_estimator (bool): Whether to add the provided estimator to the pipeline or not. Defaults to True.

    Returns:
         PipelineBase object: PipelineBase instance with dynamically generated preprocessing components and specified estimator.

    Raises:
        ValueError: If estimator is not valid for the given problem type, or sampling is not supported for the given problem type.
    """
    X = infer_feature_types(X)
    y = infer_feature_types(y)

    if estimator:
        problem_type = handle_problem_types(problem_type)
        if estimator not in get_estimators(problem_type):
            raise ValueError(
                f"{estimator.name} is not a valid estimator for problem type"
            )
        if not is_classification(problem_type) and sampler_name is not None:
            raise ValueError(
                f"Sampling is unsupported for problem_type {str(problem_type)}"
            )

    preprocessing_components = _get_preprocessing_components(
        X, y, problem_type, estimator, sampler_name
    )
    extra_components = extra_components or []
    estimator = [estimator] if use_estimator else []

    if extra_components_position == "before_preprocessing":
        complete_component_list = (
            extra_components + preprocessing_components + estimator
        )
    else:
        complete_component_list = (
            preprocessing_components + extra_components + estimator
        )

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

    Args:
        element (pipeline instance): The instance of the pipeline to generate string Python code.

    Returns:
        str: String representation of Python code that can be run separately in order to recreate the pipeline instance.
        Does not include code for custom component implementation.

    Raises:
        ValueError: If element is not a pipeline, or if the pipeline is nonlinear.
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
    input_pipelines, problem_type, final_estimator=None, n_jobs=-1, random_seed=0
):
    """Creates a pipeline with a stacked ensemble estimator.

    Args:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use as the base estimators for the stacked ensemble.
        problem_type (ProblemType): Problem type of pipeline
        final_estimator (Estimator): Metalearner to use for the ensembler. Defaults to None.
        n_jobs (int or None): Integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            Defaults to -1.

    Returns:
        Pipeline with appropriate stacked ensemble estimator.
    """

    def _make_new_component_name(model_type, component_name, idx=None):
        idx = " " + str(idx) if idx is not None else ""
        return f"{str(model_type)} Pipeline{idx} - {component_name}"

    component_graph = (
        {"Label Encoder": ["Label Encoder", "X", "y"]}
        if is_classification(problem_type)
        else {}
    )
    final_components = []
    used_model_families = []
    parameters = {}

    if is_classification(problem_type):
        parameters = {
            "Stacked Ensemble Classifier": {
                "n_jobs": n_jobs,
            }
        }
        estimator = StackedEnsembleClassifier
        pipeline_name = "Stacked Ensemble Classification Pipeline"
    else:
        parameters = {
            "Stacked Ensemble Regressor": {
                "n_jobs": n_jobs,
            }
        }
        estimator = StackedEnsembleRegressor
        pipeline_name = "Stacked Ensemble Regression Pipeline"

    pipeline_class = {
        ProblemTypes.BINARY: BinaryClassificationPipeline,
        ProblemTypes.MULTICLASS: MulticlassClassificationPipeline,
        ProblemTypes.REGRESSION: RegressionPipeline,
    }[problem_type]

    for pipeline in input_pipelines:
        model_family = pipeline.component_graph[-1].model_family
        model_family_idx = (
            used_model_families.count(model_family) + 1
            if used_model_families.count(model_family) > 0
            else None
        )
        used_model_families.append(model_family)
        final_component = None
        ensemble_y = "y"
        for name, component_list in pipeline.component_graph.component_dict.items():
            new_component_list = []
            new_component_name = _make_new_component_name(
                model_family, name, model_family_idx
            )
            for i, item in enumerate(component_list):
                if i == 0:
                    fitted_comp = handle_component_class(item)
                    new_component_list.append(fitted_comp)
                    parameters[new_component_name] = pipeline.parameters.get(name, {})
                elif isinstance(item, str) and item not in ["X", "y"]:
                    new_component_list.append(
                        _make_new_component_name(model_family, item, model_family_idx)
                    )
                elif isinstance(item, str) and item == "y":
                    if is_classification(problem_type):
                        new_component_list.append("Label Encoder.y")
                    else:
                        new_component_list.append("y")
                else:
                    new_component_list.append(item)
                if i != 0 and item.endswith(".y"):
                    ensemble_y = _make_new_component_name(
                        model_family, item, model_family_idx
                    )
            component_graph[new_component_name] = new_component_list
            final_component = new_component_name
        final_components.append(final_component)

    component_graph[estimator.name] = (
        [estimator] + [comp + ".x" for comp in final_components] + [ensemble_y]
    )

    return pipeline_class(
        component_graph,
        parameters=parameters,
        custom_name=pipeline_name,
        random_seed=random_seed,
    )


def _make_pipeline_from_multiple_graphs(
    input_pipelines,
    estimator,
    problem_type,
    parameters=None,
    pipeline_name=None,
    sub_pipeline_names=None,
    random_seed=0,
):
    """Creates a pipeline from multiple preprocessing pipelines and a final estimator. Final y input to the estimator will be chosen from the last of the input pipelines.

    Args:
        input_pipelines (list(PipelineBase or subclass obj)): List of pipeline instances to use for preprocessing.
        estimator (Estimator): Final estimator for the pipelines.
        problem_type (ProblemType): Problem type of pipeline.
        parameters (Dict): Parameters to initialize pipeline with. Defaults to an empty dictionary.
        pipeline_name (str): Custom name for the final pipeline.
        sub_pipeline_names (Dict): Dictionary mapping original input pipeline names to new names. This will be used to rename components. Defaults to None.
        random_seed (int): Random seed for the pipeline. Defaults to 0.

    Returns:
        pipeline (PipelineBase): Pipeline created with the input pipelines.
    """

    def _make_new_component_name(name, component_name, idx=None, pipeline_name=None):
        idx = " " + str(idx) if idx is not None else ""
        if pipeline_name:
            return f"{pipeline_name} Pipeline{idx} - {component_name}"
        return f"{str(name)} Pipeline{idx} - {component_name}"

    parameters = parameters if parameters else {}
    final_components = []
    used_names = []
    component_graph = (
        {"Label Encoder": ["Label Encoder", "X", "y"]}
        if is_classification(problem_type)
        else {}
    )
    for pipeline in input_pipelines:
        component_pipeline_name = pipeline.name
        name_idx = (
            used_names.count(component_pipeline_name) + 1
            if used_names.count(component_pipeline_name) > 0
            else None
        )
        used_names.append(component_pipeline_name)
        sub_pipeline_name = (
            sub_pipeline_names[pipeline.name] if sub_pipeline_names else None
        )
        final_component = None
        final_y = "y"

        final_y_candidate = (
            None
            if not handle_component_class(
                pipeline.component_graph.compute_order[-1]
            ).modifies_target
            else _make_new_component_name(
                component_pipeline_name,
                pipeline.component_graph.compute_order[-1],
                name_idx,
                sub_pipeline_name,
            )
            + ".y"
        )
        for name, component_list in pipeline.component_graph.component_dict.items():
            new_component_list = []
            new_component_name = _make_new_component_name(
                component_pipeline_name, name, name_idx, sub_pipeline_name
            )
            for i, item in enumerate(component_list):
                if i == 0:
                    fitted_comp = handle_component_class(item)
                    new_component_list.append(fitted_comp)
                    parameters[new_component_name] = pipeline.parameters.get(name, {})
                elif isinstance(item, str) and item not in ["X", "y"]:
                    new_component_list.append(
                        _make_new_component_name(
                            component_pipeline_name, item, name_idx, sub_pipeline_name
                        )
                    )
                    if i != 0 and item.endswith(".y"):
                        final_y = _make_new_component_name(
                            component_pipeline_name, item, name_idx, sub_pipeline_name
                        )
                elif isinstance(item, str) and item == "y":
                    if is_classification(problem_type):
                        new_component_list.append("Label Encoder.y")
                    else:
                        new_component_list.append("y")
                else:
                    new_component_list.append(item)
            component_graph[new_component_name] = new_component_list
            final_component = new_component_name
        final_components.append(final_component)

    final_y = final_y_candidate if final_y_candidate else final_y
    component_graph[estimator.name] = (
        [estimator] + [comp + ".x" for comp in final_components] + [final_y]
    )
    pipeline_class = {
        ProblemTypes.BINARY: BinaryClassificationPipeline,
        ProblemTypes.MULTICLASS: MulticlassClassificationPipeline,
        ProblemTypes.REGRESSION: RegressionPipeline,
        ProblemTypes.TIME_SERIES_BINARY: TimeSeriesBinaryClassificationPipeline,
        ProblemTypes.TIME_SERIES_MULTICLASS: TimeSeriesMulticlassClassificationPipeline,
        ProblemTypes.TIME_SERIES_REGRESSION: TimeSeriesRegressionPipeline,
    }[problem_type]
    return pipeline_class(
        component_graph,
        parameters=parameters,
        custom_name=pipeline_name,
        random_seed=random_seed,
    )


def make_pipeline_from_actions(problem_type, actions):
    """Creates a pipeline of components to address the input DataCheckAction list.

    Args:
        problem_type (str or ProblemType): The problem type that the pipeline should address.
        actions (list[DataCheckAction]): List of DataCheckAction objects used to create list of components

    Returns:
        PipelineBase: Pipeline which can be used to address data check actions.
    """
    component_list = _make_component_list_from_actions(actions)
    parameters = {}
    for component in component_list:
        parameters[component.name] = component.parameters
    component_dict = PipelineBase._make_component_dict_from_component_list(
        [component.name for component in component_list]
    )
    base_class = _get_pipeline_base_class(problem_type)
    return base_class(component_dict, parameters=parameters)


def _make_component_list_from_actions(actions):
    """Creates a list of components from the input DataCheckAction list.

    Args:
        actions (list(DataCheckAction)): List of DataCheckAction objects used to create list of components

    Returns:
        list(ComponentBase): List of components used to address the input actions
    """
    components = []
    cols_to_drop = []
    indices_to_drop = []
    for action in actions:
        if action.action_code == DataCheckActionCode.DROP_COL:
            cols_to_drop.extend(action.metadata["columns"])
        elif action.action_code == DataCheckActionCode.IMPUTE_COL:
            metadata = action.metadata
            if metadata["is_target"]:
                components.append(
                    TargetImputer(impute_strategy=metadata["impute_strategy"])
                )
        elif action.action_code == DataCheckActionCode.DROP_ROWS:
            indices_to_drop.extend(action.metadata["rows"])
    if cols_to_drop:
        cols_to_drop = sorted(set(cols_to_drop))
        components.append(DropColumns(columns=cols_to_drop))
    if indices_to_drop:
        indices_to_drop = sorted(set(indices_to_drop))
        components.append(DropRowsTransformer(indices_to_drop=indices_to_drop))

    return components


def make_timeseries_baseline_pipeline(problem_type, gap, forecast_horizon, date_index):
    """Make a baseline pipeline for time series regression problems.

    Args:
        problem_type: One of TIME_SERIES_REGRESSION, TIME_SERIES_MULTICLASS, TIME_SERIES_BINARY
        gap (int): Non-negative gap parameter.
        forecast_horizon (int): Positive forecast_horizon parameter.
        date_index (str): Column name of date_index parameter.

    Returns:
        TimeSeriesPipelineBase, a time series pipeline corresponding to the problem type.

    """
    pipeline_class, pipeline_name = {
        ProblemTypes.TIME_SERIES_REGRESSION: (
            TimeSeriesRegressionPipeline,
            "Time Series Baseline Regression Pipeline",
        ),
        ProblemTypes.TIME_SERIES_MULTICLASS: (
            TimeSeriesMulticlassClassificationPipeline,
            "Time Series Baseline Multiclass Pipeline",
        ),
        ProblemTypes.TIME_SERIES_BINARY: (
            TimeSeriesBinaryClassificationPipeline,
            "Time Series Baseline Binary Pipeline",
        ),
    }[problem_type]
    baseline = pipeline_class(
        component_graph=[
            "Time Series Featurizer",
            "Time Series Baseline Estimator",
        ],
        custom_name=pipeline_name,
        parameters={
            "pipeline": {
                "date_index": date_index,
                "gap": gap,
                "max_delay": 0,
                "forecast_horizon": forecast_horizon,
            },
            "Time Series Featurizer": {
                "max_delay": 0,
                "gap": gap,
                "forecast_horizon": forecast_horizon,
                "delay_target": True,
                "delay_features": False,
                "date_index": date_index,
            },
            "Time Series Baseline Estimator": {
                "gap": gap,
                "forecast_horizon": forecast_horizon,
            },
        },
    )
    return baseline


def rows_of_interest(
    pipeline, X, y=None, threshold=None, epsilon=0.1, sort_values=True, types="all"
):
    """Get the row indices of the data that are closest to the threshold. Works only for binary classification problems and pipelines.

    Args:
        pipeline (PipelineBase): The fitted binary pipeline.
        X (ww.DataTable, pd.DataFrame): The input features to predict on.
        y (ww.DataColumn, pd.Series, None): The input target data,  if available. Defaults to None.
        threshold (float): The threshold value of interest to separate positive and negative predictions. If None, uses the pipeline threshold if set, else 0.5. Defaults to None.
        epsilon (epsilon): The difference between the probability and the threshold that would make the row interesting for us. For instance, epsilon=0.1 and threhsold=0.5 would mean
            we consider all rows in [0.4, 0.6] to be of interest. Defaults to 0.1.
        sort_values (bool): Whether to return the indices sorted by the distance from the threshold, such that the first values are closer to the threshold and the later values are further. Defaults to True.
        types (str): The type of rows to keep and return. Can be one of ['incorrect', 'correct', 'true_positive', 'true_negative', 'all']. Defaults to 'all'.

            'incorrect' - return only the rows where the predictions are incorrect. This means that, given the threshold and target y, keep only the rows which are labeled wrong.
            'correct' - return only the rows where the predictions are correct. This means that, given the threshold and target y, keep only the rows which are correctly labeled.
            'true_positive' - return only the rows which are positive, as given by the targets.
            'true_negative' - return only the rows which are negative, as given by the targets.
            'all' - return all rows. This is the only option available when there is no target data provided.

    Returns:
        The indices corresponding to the rows of interest.

    Raises:
        ValueError: If pipeline is not a fitted Binary Classification pipeline.
        ValueError: If types is invalid or y is not provided when types is not 'all'.
        ValueError: If the threshold is provided and is exclusive of [0, 1].
    """
    valid_types = ["incorrect", "correct", "true_positive", "true_negative", "all"]
    if types not in valid_types:
        raise ValueError(
            "Invalid arg for 'types'! Must be one of {}".format(valid_types)
        )

    if types != "all" and y is None:
        raise ValueError("Need an input y in order to use types {}".format(types))

    if (
        not isinstance(pipeline, BinaryClassificationPipeline)
        or not pipeline._is_fitted
    ):
        raise ValueError(
            "Pipeline provided must be a fitted Binary Classification pipeline!"
        )

    if threshold is not None and (threshold < 0 or threshold > 1):
        raise ValueError(
            "Provided threshold {} must be between [0, 1]".format(threshold)
        )

    if threshold is None:
        threshold = pipeline.threshold or 0.5

    # get predicted proba
    pred_proba = pipeline.predict_proba(X)
    pos_value_proba = pred_proba.iloc[:, -1]
    preds = pos_value_proba >= threshold
    preds_value_proba = abs(pos_value_proba - threshold)

    # placeholder for y if it isn't supplied
    y_current = y if y is not None else preds

    # logic for breaking apart the different categories
    mask = y_current
    if types in ["correct", "incorrect"]:
        mask = preds == y
    mask = mask.astype(bool)

    if types in ["correct", "true_positive"]:
        preds_value_proba = preds_value_proba[mask.values]
    elif types in ["incorrect", "true_negative"]:
        preds_value_proba = preds_value_proba[~mask.values]

    if sort_values:
        preds_value_proba = preds_value_proba.sort_values(kind="stable")

    preds_value_proba = preds_value_proba[preds_value_proba <= epsilon]
    return preds_value_proba.index.tolist()
