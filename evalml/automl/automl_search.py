"""EvalML's core AutoML object."""
import copy
import logging
import pickle
import sys
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cloudpickle
import numpy as np
import pandas as pd
from dask import distributed as dd
from plotly import io as pio
from sklearn.model_selection import BaseCrossValidator
from skopt.space import Categorical

from evalml.automl.automl_algorithm import DefaultAlgorithm, IterativeAlgorithm
from evalml.automl.callbacks import log_error_callback
from evalml.automl.engine import SequentialEngine
from evalml.automl.engine.cf_engine import CFClient, CFEngine
from evalml.automl.engine.dask_engine import DaskEngine
from evalml.automl.pipeline_search_plots import PipelineSearchPlots, SearchIterationPlot
from evalml.automl.progress import Progress
from evalml.automl.utils import (
    AutoMLConfig,
    check_all_pipeline_names_unique,
    get_best_sampler_for_data,
    get_default_primary_search_objective,
    make_data_splitter,
)
from evalml.data_checks import DataCheckMessageType, DefaultDataChecks
from evalml.exceptions import (
    AutoMLSearchException,
    PipelineNotFoundError,
    PipelineScoreError,
)
from evalml.model_family import ModelFamily
from evalml.objectives import (
    get_non_core_objectives,
    get_objective,
    get_optimization_objectives,
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    ComponentGraph,
    MulticlassClassificationPipeline,
    RegressionPipeline,
)
from evalml.pipelines.components import ARIMARegressor
from evalml.pipelines.utils import make_timeseries_baseline_pipeline
from evalml.preprocessing import split_data
from evalml.problem_types import (
    ProblemTypes,
    handle_problem_types,
    is_binary,
    is_classification,
    is_time_series,
)
from evalml.tuners import SKOptTuner
from evalml.utils import convert_to_seconds, infer_feature_types
from evalml.utils.gen_utils import contains_all_ts_parameters
from evalml.utils.logger import (
    get_logger,
    log_batch_times,
    log_subtitle,
    log_title,
    time_elapsed,
)


def build_engine_from_str(engine_str):
    """Function that converts a convenience string for an parallel engine type and returns an instance of that engine.

    Args:
        engine_str (str): String representing the requested engine.

    Returns:
        (EngineBase): Instance of the requested engine.

    Raises:
        ValueError: If engine_str is not a valid engine.
    """
    valid_engines = [
        "sequential",
        "cf_threaded",
        "cf_process",
        "dask_threaded",
        "dask_process",
    ]
    if engine_str not in valid_engines:
        raise ValueError(
            f"'{engine_str}' is not a valid engine, please choose from {valid_engines}",
        )
    elif engine_str == "sequential":
        return SequentialEngine()
    elif engine_str == "cf_threaded":
        return CFEngine(CFClient(ThreadPoolExecutor()))
    elif engine_str == "cf_process":
        return CFEngine(CFClient(ProcessPoolExecutor()))
    elif engine_str == "dask_threaded":
        return DaskEngine(cluster=dd.LocalCluster(processes=False))
    elif engine_str == "dask_process":
        return DaskEngine(cluster=dd.LocalCluster(processes=True))


def search(
    X_train=None,
    y_train=None,
    problem_type=None,
    objective="auto",
    mode="fast",
    max_time=None,
    patience=None,
    tolerance=None,
    problem_configuration=None,
    n_splits=3,
    verbose=False,
    timing=False,
):
    """Given data and configuration, run an automl search.

    This method will run EvalML's default suite of data checks. If the data checks produce errors, the data check results will be returned before running the automl search. In that case we recommend you alter your data to address these errors and try again.
    This method is provided for convenience. If you'd like more control over when each of these steps is run, consider making calls directly to the various pieces like the data checks and AutoMLSearch, instead of using this method.

    Args:
        X_train (pd.DataFrame): The input training data of shape [n_samples, n_features]. Required.
        y_train (pd.Series): The target training data of length [n_samples]. Required for supervised learning tasks.
        problem_type (str or ProblemTypes): Type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.
        objective (str, ObjectiveBase): The objective to optimize for. Used to propose and rank pipelines, but not for optimizing each pipeline during fit-time.
            When set to 'auto', chooses:
            - LogLossBinary for binary classification problems,
            - LogLossMulticlass for multiclass classification problems, and
            - R2 for regression problems.
        mode (str): mode for DefaultAlgorithm. There are two modes: fast and long, where fast is a subset of long. Please look at DefaultAlgorithm for more details.
        max_time (int, str): Maximum time to search for pipelines.
            This will not start a new pipeline search after the duration
            has elapsed. If it is an integer, then the time will be in seconds.
            For strings, time can be specified as seconds, minutes, or hours.
        patience (int): Number of iterations without improvement to stop search early. Must be positive.
            If None, early stopping is disabled. Defaults to None.
        tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
            Only applicable if patience is not None. Defaults to None.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, forecast_horizon, and max_delay variables.
        n_splits (int): Number of splits to use with the default data splitter.
        verbose (boolean): Whether or not to display semi-real-time updates to stdout while search is running. Defaults to False.
        timing (boolean): Whether or not to write pipeline search times to the logger. Defaults to False.

    Returns:
        (AutoMLSearch, dict): The automl search object containing pipelines and rankings, and the results from running the data checks. If the data check results contain errors, automl search will not be run and an automl search object will not be returned.

    Raises:
        ValueError: If search configuration is not valid.
    """
    X_train = infer_feature_types(X_train)
    y_train = infer_feature_types(y_train)
    problem_type = handle_problem_types(problem_type)

    if is_time_series(problem_type):
        is_valid, msg = contains_all_ts_parameters(problem_configuration)
        if not is_valid:
            raise ValueError(msg)

    if objective == "auto":
        objective = get_default_primary_search_objective(problem_type)
    objective = get_objective(objective, return_instance=False)

    if mode != "fast" and mode != "long":
        raise ValueError("Mode must be either 'fast' or 'long'")

    max_batches = None
    if mode == "fast":
        max_batches = 4  # corresponds to end of 'fast' mode
    elif mode == "long" and max_time:
        max_batches = 999  # defers to stopping criterion
    elif mode == "long" and max_time is None:
        max_batches = 6  # corresponds to end of 'long' exploration phase

    data_splitter = make_data_splitter(
        X=X_train,
        y=y_train,
        problem_type=problem_type,
        problem_configuration=problem_configuration,
        n_splits=n_splits,
    )

    automl_config = {
        "X_train": X_train,
        "y_train": y_train,
        "problem_type": problem_type,
        "objective": objective,
        "max_batches": max_batches,
        "max_time": max_time,
        "patience": patience,
        "tolerance": tolerance,
        "verbose": verbose,
        "problem_configuration": problem_configuration,
        "data_splitter": data_splitter,
        "timing": timing,
    }

    data_checks = DefaultDataChecks(
        problem_type=problem_type,
        objective=objective,
        n_splits=n_splits,
        problem_configuration=problem_configuration,
    )
    data_check_results = data_checks.validate(X_train, y=y_train)

    for data_check_result in data_check_results:
        if data_check_result["level"] == DataCheckMessageType.ERROR.value:
            return None, data_check_results

    automl = AutoMLSearch(automl_algorithm="default", ensembling=True, **automl_config)
    automl.search()
    return automl, data_check_results


def search_iterative(
    X_train=None,
    y_train=None,
    problem_type=None,
    objective="auto",
    problem_configuration=None,
    n_splits=3,
    timing=False,
    **kwargs,
):
    """Given data and configuration, run an automl search.

    This method will run EvalML's default suite of data checks. If the data checks produce errors, the data check results will be returned before running the automl search. In that case we recommend you alter your data to address these errors and try again.
    This method is provided for convenience. If you'd like more control over when each of these steps is run, consider making calls directly to the various pieces like the data checks and AutoMLSearch, instead of using this method.

    Args:
        X_train (pd.DataFrame): The input training data of shape [n_samples, n_features]. Required.
        y_train (pd.Series): The target training data of length [n_samples]. Required for supervised learning tasks.
        problem_type (str or ProblemTypes): Type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.
        objective (str, ObjectiveBase): The objective to optimize for. Used to propose and rank pipelines, but not for optimizing each pipeline during fit-time.
            When set to 'auto', chooses:
            - LogLossBinary for binary classification problems,
            - LogLossMulticlass for multiclass classification problems, and
            - R2 for regression problems.
        problem_configuration (dict): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, forecast_horizon, and max_delay variables.
        n_splits (int): Number of splits to use with the default data splitter.
        timing(boolean): Whether or not to write pipeline search times to the logger. Defaults to False.
        **kwargs: Other keyword arguments which are provided will be passed to AutoMLSearch.

    Returns:
        (AutoMLSearch, dict): the automl search object containing pipelines and rankings, and the results from running the data checks. If the data check results contain errors, automl search will not be run and an automl search object will not be returned.

    Raises:
        ValueError: If the search configuration is invalid.
    """
    X_train = infer_feature_types(X_train)
    y_train = infer_feature_types(y_train)
    problem_type = handle_problem_types(problem_type)

    if is_time_series(problem_type):
        is_valid, msg = contains_all_ts_parameters(problem_configuration)
        if not is_valid:
            raise ValueError(msg)

    if objective == "auto":
        objective = get_default_primary_search_objective(problem_type)
    objective = get_objective(objective, return_instance=False)

    data_splitter = make_data_splitter(
        X=X_train,
        y=y_train,
        problem_type=problem_type,
        problem_configuration=problem_configuration,
        n_splits=n_splits,
    )

    automl_config = kwargs
    automl_config.update(
        {
            "X_train": X_train,
            "y_train": y_train,
            "problem_type": problem_type,
            "objective": objective,
            "max_batches": 1,
            "problem_configuration": problem_configuration,
            "data_splitter": data_splitter,
            "timing": timing,
        },
    )

    data_checks = DefaultDataChecks(
        problem_type=problem_type,
        objective=objective,
        n_splits=n_splits,
        problem_configuration=problem_configuration,
    )
    data_check_results = data_checks.validate(X_train, y=y_train)
    for data_check_result in data_check_results:
        if data_check_result["level"] == DataCheckMessageType.ERROR.value:
            return None, data_check_results

    automl = AutoMLSearch(**automl_config)
    automl.search()
    return automl, data_check_results


class AutoMLSearch:
    """Automated Pipeline search.

    Args:
        X_train (pd.DataFrame): The input training data of shape [n_samples, n_features]. Required.

        y_train (pd.Series): The target training data of length [n_samples]. Required for supervised learning tasks.

        X_holdout (pd.DataFrame): The input holdout data of shape [n_samples, n_features].

        y_holdout (pd.Series): The target holdout data of length [n_samples].

        problem_type (str or ProblemTypes): Type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

        objective (str, ObjectiveBase): The objective to optimize for. Used to propose and rank pipelines, but not for optimizing each pipeline during fit-time.
            When set to 'auto', chooses:
            - LogLossBinary for binary classification problems,
            - LogLossMulticlass for multiclass classification problems, and
            - R2 for regression problems.

        max_iterations (int): Maximum number of iterations to search. If max_iterations and
            max_time is not set, then max_iterations will default to max_iterations of 5.

        max_time (int, str): Maximum time to search for pipelines.
            This will not start a new pipeline search after the duration
            has elapsed. If it is an integer, then the time will be in seconds.
            For strings, time can be specified as seconds, minutes, or hours.

        patience (int): Number of iterations without improvement to stop search early. Must be positive.
            If None, early stopping is disabled. Defaults to None.

        tolerance (float): Minimum percentage difference to qualify as score improvement for early stopping.
            Only applicable if patience is not None. Defaults to None.

        allowed_component_graphs (dict): A dictionary of lists or ComponentGraphs indicating the component graphs allowed in the search.
            The format should follow { "Name_0": [list_of_components], "Name_1": ComponentGraph(...) }

            The default of None indicates all pipeline component graphs for this problem type are allowed. Setting this field will cause
            allowed_model_families to be ignored.

            e.g. allowed_component_graphs = { "My_Graph": ["Imputer", "One Hot Encoder", "Random Forest Classifier"] }

        allowed_model_families (list(str, ModelFamily)): The model families to search. The default of None searches over all
            model families. Run evalml.pipelines.components.utils.allowed_model_families("binary") to see options. Change `binary`
            to `multiclass` or `regression` depending on the problem type. Note that if allowed_pipelines is provided,
            this parameter will be ignored.

        features (list)[FeatureBase]: List of features to run DFS on AutoML pipelines. Defaults to None. Features will only be computed if the columns used by the feature exist in the search input and if the feature itself is not in search input.

        data_splitter (sklearn.model_selection.BaseCrossValidator): Data splitting method to use. Defaults to StratifiedKFold.

        tuner_class: The tuner class to use. Defaults to SKOptTuner.

        optimize_thresholds (bool): Whether or not to optimize the binary pipeline threshold. Defaults to True.

        start_iteration_callback (callable): Function called before each pipeline training iteration.
            Callback function takes three positional parameters: The pipeline instance and the AutoMLSearch object.

        add_result_callback (callable): Function called after each pipeline training iteration.
            Callback function takes three positional parameters: A dictionary containing the training results for the new pipeline, an untrained_pipeline containing the parameters used during training, and the AutoMLSearch object.

        error_callback (callable): Function called when `search()` errors and raises an Exception.
            Callback function takes three positional parameters: the Exception raised, the traceback, and the AutoMLSearch object.
            Must also accepts kwargs, so AutoMLSearch is able to pass along other appropriate parameters by default.
            Defaults to None, which will call `log_error_callback`.

        additional_objectives (list): Custom set of objectives to score on.
            Will override default objectives for problem type if not empty.

        alternate_thresholding_objective (str): The objective to use for thresholding binary classification pipelines if the main objective provided isn't tuneable.
            Defaults to F1.

        random_seed (int): Seed for the random number generator. Defaults to 0.

        n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
            None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

        ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over.
            If the number of unique pipelines to search over per batch is one, ensembling will not run. Defaults to False.

        max_batches (int): The maximum number of batches of pipelines to search. Parameters max_time, and
            max_iterations have precedence over stopping the search.

        problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
            in time series problems, values should be passed in for the time_index, gap, forecast_horizon, and max_delay variables.

        train_best_pipeline (boolean): Whether or not to train the best pipeline before returning it. Defaults to True.

        search_parameters (dict): A dict of the hyperparameter ranges or pipeline parameters used to iterate over during search.
            Keys should consist of the component names and values should specify a singular value/list for pipeline parameters, or skopt.Space for hyperparameter ranges.
            In the example below, the Imputer parameters would be passed to the hyperparameter ranges, and the Label Encoder parameters would be used as the component parameter.

            e.g. search_parameters = { 'Imputer' : { 'numeric_impute_strategy': Categorical(['most_frequent', 'median']) },
                                       'Label Encoder': {'positive_label': True} }

        sampler_method (str): The data sampling component to use in the pipelines if the problem type is classification and the target balance is smaller than the sampler_balanced_ratio.
            Either 'auto', which will use our preferred sampler for the data, 'Undersampler', 'Oversampler', or None. Defaults to 'auto'.

        sampler_balanced_ratio (float): The minority:majority class ratio that we consider balanced, so a 1:4 ratio would be equal to 0.25. If the class balance is larger than this provided value,
            then we will not add a sampler since the data is then considered balanced. Overrides the `sampler_ratio` of the samplers. Defaults to 0.25.

        allow_long_running_models (bool): Whether or not to allow longer-running models for large multiclass problems. If False and no pipelines, component graphs, or model families are provided,
            AutoMLSearch will not use Elastic Net or XGBoost when there are more than 75 multiclass targets and will not use CatBoost when there are more than 150 multiclass targets. Defaults to False.

        _ensembling_split_size (float): The amount of the training data we'll set aside for training ensemble metalearners. Only used when ensembling is True.
            Must be between 0 and 1, exclusive. Defaults to 0.2

        _pipelines_per_batch (int): The number of pipelines to train for every batch after the first one.
            The first batch will train a baseline pipline + one of each pipeline family allowed in the search.

        automl_algorithm (str): The automl algorithm to use. Currently the two choices are 'iterative' and 'default'. Defaults to `default`.

        engine (EngineBase or str): The engine instance used to evaluate pipelines. Dask or concurrent.futures engines can also
            be chosen by providing a string from the list ["sequential", "cf_threaded", "cf_process", "dask_threaded", "dask_process"].
            If a parallel engine is selected this way, the maximum amount of parallelism, as determined by the engine, will be used. Defaults to "sequential".

        verbose (boolean): Whether or not to display semi-real-time updates to stdout while search is running. Defaults to False.

        timing (boolean): Whether or not to write pipeline search times to the logger. Defaults to False.
        exclude_featurizers (list[str]): A list of featurizer components to exclude from the pipelines built by search.
            Valid options are "DatetimeFeaturizer", "EmailFeaturizer", "URLFeaturizer", "NaturalLanguageFeaturizer", "TimeSeriesFeaturizer"

        holdout_set_size (float): The size of the holdout set that AutoML search will take for datasets larger than 500 rows. If set to 0, holdout set will not be taken regardless of number of rows. Must be between 0 and 1, exclusive. Defaults to 0.1.
    """

    _MAX_NAME_LEN = 40

    # Minimum number of rows dataset must have before a holdout set is used to rank pipelines.
    _HOLDOUT_SET_MIN_ROWS = 500

    def __init__(
        self,
        X_train=None,
        y_train=None,
        X_holdout=None,
        y_holdout=None,
        problem_type=None,
        objective="auto",
        max_iterations=None,
        max_time=None,
        patience=None,
        tolerance=None,
        data_splitter=None,
        allowed_component_graphs=None,
        allowed_model_families=None,
        features=None,
        start_iteration_callback=None,
        add_result_callback=None,
        error_callback=None,
        additional_objectives=None,
        alternate_thresholding_objective="F1",
        random_seed=0,
        n_jobs=-1,
        tuner_class=None,
        optimize_thresholds=True,
        ensembling=False,
        max_batches=None,
        problem_configuration=None,
        train_best_pipeline=True,
        search_parameters=None,
        sampler_method="auto",
        sampler_balanced_ratio=0.25,
        allow_long_running_models=False,
        _pipelines_per_batch=5,
        automl_algorithm="default",
        engine="sequential",
        verbose=False,
        timing=False,
        exclude_featurizers=None,
        holdout_set_size=0,
    ):
        self.verbose = verbose
        if verbose:
            self.logger = get_logger(f"{__name__}.verbose")
        else:
            self.logger = logging.getLogger(__name__)
        self.timing = timing

        if X_train is None:
            raise ValueError(
                "Must specify training data as a 2d array using the X_train argument",
            )
        if y_train is None:
            raise ValueError(
                "Must specify training data target values as a 1d vector using the y_train argument",
            )

        if X_holdout is not None and y_holdout is not None:
            self.passed_holdout_set = True
        elif X_holdout is None and y_holdout is None:
            self.passed_holdout_set = False
        elif X_holdout is None and y_holdout is not None:
            raise ValueError(
                "Must specify holdout data as a 2d array using the X_holdout argument",
            )
        elif X_holdout is not None and y_holdout is None:
            raise ValueError(
                "Must specify training data target values as a 1d vector using the y_holdout argument",
            )

        try:
            self.problem_type = handle_problem_types(problem_type)
        except ValueError:
            raise ValueError(
                "choose one of (binary, multiclass, regression) as problem_type",
            )

        if is_time_series(self.problem_type):
            warnings.warn(
                "Time series support in evalml is still in beta, which means we are still actively building "
                "its core features. Please be mindful of that when running search().",
            )
        self.errors = {}
        self._SLEEP_TIME = 0.1
        self.tuner_class = tuner_class or SKOptTuner
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.error_callback = error_callback or log_error_callback
        self.data_splitter = data_splitter
        self.optimize_thresholds = optimize_thresholds
        self.ensembling = ensembling
        if objective == "auto":
            objective = get_default_primary_search_objective(self.problem_type.value)
        objective = get_objective(objective, return_instance=False)
        self.objective = self._validate_objective(objective)
        self.alternate_thresholding_objective = None
        if (
            is_binary(self.problem_type)
            and self.optimize_thresholds
            and self.objective.score_needs_proba
        ):
            self.alternate_thresholding_objective = get_objective(
                alternate_thresholding_objective,
                return_instance=True,
            )
        if (
            self.alternate_thresholding_objective is not None
            and self.alternate_thresholding_objective.score_needs_proba
        ):
            raise ValueError(
                "Alternate thresholding objective must be a tuneable objective and cannot need probabilities!",
            )
        if self.data_splitter is not None and not issubclass(
            self.data_splitter.__class__,
            BaseCrossValidator,
        ):
            raise ValueError("Not a valid data splitter")
        if not objective.is_defined_for_problem_type(self.problem_type):
            raise ValueError(
                "Given objective {} is not compatible with a {} problem.".format(
                    self.objective.name,
                    self.problem_type.value,
                ),
            )
        if additional_objectives is None:
            additional_objectives = get_optimization_objectives(self.problem_type)
            # if our main objective is part of default set of objectives for problem_type, remove it
            existing_main_objective = next(
                (
                    obj
                    for obj in additional_objectives
                    if obj.name == self.objective.name
                ),
                None,
            )
            if existing_main_objective is not None:
                additional_objectives.remove(existing_main_objective)
        else:
            additional_objectives = [get_objective(o) for o in additional_objectives]
        additional_objectives = [
            self._validate_objective(obj) for obj in additional_objectives
        ]
        self.additional_objectives = additional_objectives
        self.objective_name_to_class = {
            o.name: o for o in [self.objective] + self.additional_objectives
        }

        if not isinstance(max_time, (int, float, str, type(None))):
            raise TypeError(
                f"Parameter max_time must be a float, int, string or None. Received {type(max_time)} with value {str(max_time)}..",
            )
        if isinstance(max_time, (int, float)) and max_time < 0:
            raise ValueError(
                f"Parameter max_time must be None or non-negative. Received {max_time}.",
            )
        if max_batches is not None and max_batches < 0:
            raise ValueError(
                f"Parameter max_batches must be None or non-negative. Received {max_batches}.",
            )
        if max_iterations is not None and max_iterations < 0:
            raise ValueError(
                f"Parameter max_iterations must be None or non-negative. Received {max_iterations}.",
            )
        self.max_time = (
            convert_to_seconds(max_time) if isinstance(max_time, str) else max_time
        )
        self.max_iterations = max_iterations
        self.max_batches = max_batches
        self._pipelines_per_batch = _pipelines_per_batch
        self.holdout_set_size = holdout_set_size

        if patience and (not isinstance(patience, int) or patience < 0):
            raise ValueError(
                "patience value must be a positive integer. Received {} instead".format(
                    patience,
                ),
            )

        if tolerance and (tolerance > 1.0 or tolerance < 0.0):
            raise ValueError(
                "tolerance value must be a float between 0.0 and 1.0 inclusive. Received {} instead".format(
                    tolerance,
                ),
            )

        self.patience = patience
        self.tolerance = tolerance or 0.0

        self._results = {
            "pipeline_results": {},
            "search_order": [],
        }
        self._pipelines_searched = dict()
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        if allowed_component_graphs is not None:
            if not isinstance(allowed_component_graphs, dict):
                raise ValueError(
                    "Parameter allowed_component_graphs must be either None or a dictionary!",
                )
            for graph_name, graph in allowed_component_graphs.items():
                if not isinstance(graph, (list, dict, ComponentGraph)):
                    raise ValueError(
                        "Every component graph passed must be of type list, dictionary, or ComponentGraph!",
                    )
        self.allowed_component_graphs = allowed_component_graphs
        self.allowed_model_families = allowed_model_families
        self.allow_long_running_models = allow_long_running_models
        self._start = 0.0
        self._baseline_cv_scores = {}
        self.show_batch_output = False

        self._validate_problem_type()
        self.problem_configuration = self._validate_problem_configuration(
            problem_configuration,
        )
        self._train_best_pipeline = train_best_pipeline
        self._best_pipeline = None
        self._searched = False

        if self.holdout_set_size < 0 or holdout_set_size >= 1:
            raise ValueError(
                "Holdout set size must be greater than 0 and less than 1. Set holdout set size to 0 to disable holdout set evaluation.",
            )
        if self.passed_holdout_set is False and self.holdout_set_size > 0:
            if len(X_train) >= self._HOLDOUT_SET_MIN_ROWS:
                # Create holdout set from X_train and y_train data because X_train above or at row threshold
                X_train, X_holdout, y_train, y_holdout = split_data(
                    X_train,
                    y_train,
                    problem_type=self.problem_type,
                    problem_configuration=self.problem_configuration,
                    test_size=self.holdout_set_size,
                    random_seed=self.random_seed,
                )
                self.logger.info(
                    f"Created a holdout dataset with {len(X_holdout)} rows. Training dataset has {len(X_train)} rows.",
                )
            else:
                self.logger.info(
                    f"Dataset size is too small to create holdout set. Minimum dataset size is {self._HOLDOUT_SET_MIN_ROWS} rows, X_train has {len(X_train)} rows. Holdout set evaluation is disabled.",
                )
        # Set holdout data in AutoML search if provided as parameter
        self.X_train = infer_feature_types(X_train)
        self.y_train = infer_feature_types(y_train)
        self.X_holdout = (
            infer_feature_types(X_holdout) if X_holdout is not None else None
        )
        self.y_holdout = (
            infer_feature_types(y_holdout) if y_holdout is not None else None
        )
        if self.X_holdout is None and self.y_holdout is None:
            # Holdout set enabled but not enough rows
            self.logger.info(
                f"AutoMLSearch will use mean CV score to rank pipelines.",
            )
        else:
            self.logger.info(
                "AutoMLSearch will use the holdout set to score and rank pipelines.",
            )

        default_data_splitter = make_data_splitter(
            self.X_train,
            self.y_train,
            self.problem_type,
            self.problem_configuration,
            n_splits=3,
            shuffle=True,
            random_seed=self.random_seed,
        )
        self.data_splitter = self.data_splitter or default_data_splitter
        self.search_parameters = search_parameters or {}
        # Fitting takes a long time if the data is too wide or long.
        if is_time_series(problem_type) and (
            self.X_train.shape[1] >= 10 or self.X_train.shape[0] >= 10000
        ):
            user_arima_hyperparams = ARIMARegressor.name in self.search_parameters
            if user_arima_hyperparams and not self.search_parameters[
                ARIMARegressor.name
            ].get("use_covariates"):
                self.search_parameters[ARIMARegressor.name].update(
                    {"use_covariates": Categorical([False])},
                )
            elif not user_arima_hyperparams:
                self.search_parameters[ARIMARegressor.name] = {
                    "use_covariates": Categorical([False]),
                }

        self.search_iteration_plot = None
        self._interrupted = False
        internal_search_parameters = copy.copy(self.search_parameters)

        if self.problem_configuration:
            internal_search_parameters.update({"pipeline": self.problem_configuration})

        self.features = features
        if self.features is not None:
            internal_search_parameters.update(
                {"DFS Transformer": {"features": self.features}},
            )

        self.sampler_method = sampler_method
        self.sampler_balanced_ratio = sampler_balanced_ratio
        self._sampler_name = None

        featurizer_names = [
            "DatetimeFeaturizer",
            "EmailFeaturizer",
            "URLFeaturizer",
            "NaturalLanguageFeaturizer",
            "TimeSeriesFeaturizer",
        ]
        if exclude_featurizers and (set(exclude_featurizers) - set(featurizer_names)):
            raise ValueError(
                f"Invalid value provided for exclude_featurizers. Must be one of: {', '.join(featurizer_names)}",
            )

        if exclude_featurizers and is_time_series(problem_type):
            if (
                "DatetimeFeaturizer" in exclude_featurizers
                and "TimeSeriesFeaturizer" not in exclude_featurizers
            ):
                raise ValueError(
                    "For time series problems, if DatetimeFeaturizer is excluded, must also exclude TimeSeriesFeaturizer",
                )
            elif (
                "TimeSeriesFeaturizer" in exclude_featurizers
                and "DatetimeFeaturizer" not in exclude_featurizers
            ):
                raise ValueError(
                    "For time series problems, if TimeSeriesFeaturizer is excluded, must also exclude DatetimeFeaturizer",
                )
        self.exclude_featurizers = exclude_featurizers or []

        if is_classification(self.problem_type):
            self._sampler_name = self.sampler_method
            if self.sampler_method == "auto":
                self._sampler_name = get_best_sampler_for_data(
                    self.X_train,
                    self.y_train,
                    self.sampler_method,
                    self.sampler_balanced_ratio,
                )
            if (
                self._sampler_name not in internal_search_parameters
                and self._sampler_name is not None
            ):
                internal_search_parameters[self._sampler_name] = {
                    "sampling_ratio": self.sampler_balanced_ratio,
                }
            elif self._sampler_name is not None:
                internal_search_parameters[self._sampler_name].update(
                    {"sampling_ratio": self.sampler_balanced_ratio},
                )

        if isinstance(engine, str):
            self._engine = build_engine_from_str(engine)
        elif isinstance(engine, (DaskEngine, CFEngine, SequentialEngine)):
            self._engine = engine
        else:
            raise TypeError(
                "Invalid type provided for 'engine'.  Requires string, DaskEngine instance, or CFEngine instance.",
            )

        self.automl_config = AutoMLConfig(
            self.data_splitter,
            self.problem_type,
            self.objective,
            self.additional_objectives,
            self.alternate_thresholding_objective,
            self.optimize_thresholds,
            self.error_callback,
            self.random_seed,
            self.X_train.ww.schema,
            self.y_train.ww.schema,
            self.errors,
        )

        text_in_ensembling = (
            len(self.X_train.ww.select("natural_language", return_schema=True).columns)
            > 0
        )

        if automl_algorithm == "iterative":
            self.automl_algorithm = IterativeAlgorithm(
                X=self.X_train,
                y=self.y_train,
                problem_type=self.problem_type,
                sampler_name=self._sampler_name,
                allowed_component_graphs=self.allowed_component_graphs,
                allowed_model_families=self.allowed_model_families,
                max_iterations=self.max_iterations,
                max_batches=self.max_batches,
                tuner_class=self.tuner_class,
                random_seed=self.random_seed,
                n_jobs=self.n_jobs,
                number_features=self.X_train.shape[1],
                pipelines_per_batch=self._pipelines_per_batch,
                ensembling=self.ensembling,
                text_in_ensembling=text_in_ensembling,
                search_parameters=internal_search_parameters,
                allow_long_running_models=allow_long_running_models,
                features=features,
                verbose=self.verbose,
                exclude_featurizers=self.exclude_featurizers,
            )
        elif automl_algorithm == "default":
            self.automl_algorithm = DefaultAlgorithm(
                X=self.X_train,
                y=self.y_train,
                problem_type=self.problem_type,
                sampler_name=self._sampler_name,
                tuner_class=self.tuner_class,
                random_seed=self.random_seed,
                search_parameters=internal_search_parameters,
                text_in_ensembling=text_in_ensembling,
                allow_long_running_models=allow_long_running_models,
                features=features,
                ensembling=self.ensembling,
                verbose=self.verbose,
                n_jobs=self.n_jobs,
                exclude_featurizers=self.exclude_featurizers,
            )
        else:
            raise ValueError("Please specify a valid automl algorithm.")

        self.allowed_pipelines = self.automl_algorithm.allowed_pipelines
        self.allowed_model_families = [p.model_family for p in self.allowed_pipelines]
        if automl_algorithm == "iterative":
            self.max_iterations = self.automl_algorithm.max_iterations

        if not self.max_iterations and not self.max_time and not self.max_batches:
            self.max_batches = self.automl_algorithm.default_max_batches
            self.logger.info(
                f"Using default limit of max_batches={self.max_batches}.\n",
            )

        self.progress = Progress(
            max_time=self.max_time,
            max_batches=self.max_batches,
            max_iterations=self.max_iterations,
            patience=self.patience,
            tolerance=self.tolerance,
            automl_algorithm=self.automl_algorithm,
            objective=self.objective,
            verbose=verbose,
        )

    def close_engine(self):
        """Function to explicitly close the engine, client, parallel resources."""
        self._engine.close()

    def _get_batch_number(self):
        batch_number = 0
        if self.automl_algorithm is not None and self.automl_algorithm.batch_number > 0:
            batch_number = self.automl_algorithm.batch_number
        return batch_number

    def _pre_evaluation_callback(self, pipeline):
        if self.start_iteration_callback:
            self.start_iteration_callback(pipeline, self)

    def _validate_objective(self, objective):
        non_core_objectives = get_non_core_objectives()
        if isinstance(objective, type):
            if objective in non_core_objectives:
                raise ValueError(
                    f"{objective.name.lower()} is not allowed in AutoML! "
                    "Use evalml.objectives.utils.get_optimization_objectives() "
                    "to get all objectives allowed for automl optimization.",
                )
            return objective()
        return objective

    def __str__(self):
        """Returns string representation of the AutoMLSearch object."""

        def _print_list(obj_list):
            lines = sorted(["\t{}".format(o.name) for o in obj_list])
            return "\n".join(lines)

        def _get_funct_name(function):
            if callable(function):
                return function.__name__
            else:
                return None

        search_desc = (
            f"{handle_problem_types(self.problem_type).name} Search\n\n"
            f"Parameters: \n{'='*20}\n"
            f"Objective: {get_objective(self.objective).name}\n"
            f"Max Time: {self.max_time}\n"
            f"Max Iterations: {self.max_iterations}\n"
            f"Max Batches: {self.max_batches}\n"
            f"Allowed Pipelines: \n{_print_list(self.allowed_pipelines or [])}\n"
            f"Patience: {self.patience}\n"
            f"Tolerance: {self.tolerance}\n"
            f"Data Splitting: {self.data_splitter}\n"
            f"Tuner: {self.tuner_class.__name__}\n"
            f"Start Iteration Callback: {_get_funct_name(self.start_iteration_callback)}\n"
            f"Add Result Callback: {_get_funct_name(self.add_result_callback)}\n"
            f"Additional Objectives: {_print_list(self.additional_objectives or [])}\n"
            f"Random Seed: {self.random_seed}\n"
            f"n_jobs: {self.n_jobs}\n"
            f"Optimize Thresholds: {self.optimize_thresholds}\n"
        )

        rankings_desc = ""
        if not self.rankings.empty:
            rankings_str = self.rankings.drop(
                ["parameters"],
                axis="columns",
            ).to_string()
            rankings_desc = f"\nSearch Results: \n{'='*20}\n{rankings_str}"

        return search_desc + rankings_desc

    def _validate_problem_configuration(self, problem_configuration=None):
        if is_time_series(self.problem_type):
            is_valid, msg = contains_all_ts_parameters(problem_configuration)
            if not is_valid:
                raise ValueError(msg)
        return problem_configuration or {}

    def _handle_keyboard_interrupt(self):
        """Presents a prompt to the user asking if they want to stop the search.

        Returns:
            bool: If True, search should terminate early.
        """
        leading_char = "\n"
        start_of_loop = time.time()
        while True:
            choice = (
                input(leading_char + "Do you really want to exit search (y/n)? ")
                .strip()
                .lower()
            )
            if choice == "y":
                self.logger.info("Exiting AutoMLSearch.")
                return True
            elif choice == "n":
                # So that the time in this loop does not count towards the time budget (if set)
                time_in_loop = time.time() - start_of_loop
                self.progress.start_time += time_in_loop
                return False
            else:
                leading_char = ""

    def search(self, interactive_plot=True):
        """Find the best pipeline for the data set.

        Args:
            interactive_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.

        Raises:
            AutoMLSearchException: If all pipelines in the current AutoML batch produced a score of np.nan on the primary objective.

        Returns:
            Dict[int, Dict[str, Timestamp]]: Dictionary keyed by batch number that maps to the timings for pipelines run in that batch,
            as well as the total time for each batch. Pipelines within a batch are labeled by pipeline name.
        """
        batch_times = {}

        if self._searched:
            self.logger.error(
                "AutoMLSearch.search() has already been run and will not run again on the same instance. Re-initialize AutoMLSearch to search again.",
            )
            return

        # don't show iteration plot outside of a jupyter notebook
        if interactive_plot:
            try:
                get_ipython
            except NameError:
                interactive_plot = False

        log_title(self.logger, "Beginning pipeline search")
        self.logger.info("Optimizing for %s. " % self.objective.name)
        self.logger.info(
            "{} score is better.\n".format(
                "Greater" if self.objective.greater_is_better else "Lower",
            ),
        )
        self.logger.info(
            f"Using {self._engine.__class__.__name__} to train and score pipelines.",
        )

        if self.max_batches is not None:
            self.logger.info(
                f"Searching up to {self.max_batches} batches for a total of {self.max_iterations} pipelines. ",
            )
        elif self.max_iterations is not None:
            self.logger.info("Searching up to %s pipelines. " % self.max_iterations)
        if self.max_time is not None:
            self.logger.info(
                "Will stop searching for new pipelines after %d seconds.\n"
                % self.max_time,
            )
        self.logger.info(
            "Allowed model families: %s\n"
            % ", ".join([model.value for model in self.allowed_model_families]),
        )
        self.search_iteration_plot = None
        if self.plot and self.verbose:
            self.search_iteration_plot = self.plot.search_iteration_plot(
                interactive_plot=interactive_plot,
            )

        self.progress.start_timing()

        try:
            self._add_baseline_pipelines()
        except KeyboardInterrupt:
            if self._handle_keyboard_interrupt():
                self._interrupted = True

        current_batch_pipelines = []
        current_batch_pipeline_scores = []
        new_pipeline_ids = []
        loop_interrupted = False

        while self.progress.should_continue(
            results=self._results,
            interrupted=self._interrupted,
        ):
            pipeline_times = {}
            start_batch_time = time.time()
            computations = []
            try:
                if not loop_interrupted:
                    current_batch_pipelines = self.automl_algorithm.next_batch()
            except StopIteration:
                self.logger.info("AutoML Algorithm out of recommendations, ending")
                break
            try:
                if self.progress.should_continue(
                    results=self._results,
                    interrupted=self._interrupted,
                    mid_batch=True,
                ):
                    new_pipeline_ids = []
                    log_title(
                        self.logger,
                        f"Evaluating Batch Number {self._get_batch_number()}",
                    )
                    for pipeline in current_batch_pipelines:
                        self._pre_evaluation_callback(pipeline)
                        computation = self._engine.submit_evaluation_job(
                            self.automl_config,
                            pipeline,
                            self.X_train,
                            self.y_train,
                            self.X_holdout,
                            self.y_holdout,
                        )
                        computations.append((computation, False))
                    current_computation_index = 0
                    computations_left_to_process = len(computations)
                while (
                    self.progress.should_continue(
                        results=self._results,
                        interrupted=self._interrupted,
                        mid_batch=True,
                    )
                    and computations_left_to_process > 0
                ):
                    computation, has_been_processed = computations[
                        current_computation_index
                    ]
                    if computation.done() and not has_been_processed:
                        start_pipeline_time = time.time()
                        evaluation = computation.get_result()
                        data, cached_data, pipeline, job_log = (
                            evaluation.get("scores"),
                            evaluation.get("cached_data"),
                            evaluation.get("pipeline"),
                            evaluation.get("logger"),
                        )
                        pipeline_id = self._post_evaluation_callback(
                            pipeline,
                            data,
                            cached_data,
                            job_log,
                        )
                        pipeline_times[pipeline.name] = time_elapsed(
                            start_pipeline_time,
                        )
                        new_pipeline_ids.append(pipeline_id)
                        computations[current_computation_index] = (computation, True)
                        computations_left_to_process -= 1
                    current_computation_index = (current_computation_index + 1) % max(
                        len(computations),
                        1,
                    )
                    time.sleep(self._sleep_time)
                loop_interrupted = False
            except KeyboardInterrupt:
                loop_interrupted = True
                if self._handle_keyboard_interrupt():
                    self._interrupted = True
                    for computation, has_been_processed in computations:
                        if not has_been_processed:
                            computation.cancel()

            full_rankings = self.full_rankings
            current_batch_idx = full_rankings["id"].isin(new_pipeline_ids)
            current_batch_pipeline_scores = full_rankings[current_batch_idx][
                "validation_score"
            ]

            if (
                len(current_batch_pipeline_scores)
                and current_batch_pipeline_scores.isna().all()
            ):
                error_msgs = set(
                    [str(pl_fold["Exception"]) for pl_fold in self.errors.values()],
                )
                raise AutoMLSearchException(
                    f"All pipelines in the current AutoML batch produced a score of np.nan on the primary objective {self.objective}. Exception(s) raised: {error_msgs}. Check the 'errors' attribute of the AutoMLSearch object for a full breakdown of errors and tracebacks.",
                )
            if len(pipeline_times) > 0:
                pipeline_times["Total time of batch"] = time_elapsed(start_batch_time)
                batch_times[self._get_batch_number()] = pipeline_times

        self.search_duration = time.time() - self.progress.start_time
        elapsed_time = time_elapsed(self.progress.start_time)
        desc = f"\nSearch finished after {elapsed_time}"
        desc = desc.ljust(self._MAX_NAME_LEN)
        self.logger.info(desc)

        if self.timing is True:
            log_batch_times(self.logger, batch_times)

        self._find_best_pipeline()
        if self._best_pipeline is not None:
            best_pipeline = self.rankings.iloc[0]
            best_pipeline_name = best_pipeline["pipeline_name"]
            self.logger.info(f"Best pipeline: {best_pipeline_name}")
            self.logger.info(
                f"Best pipeline {self.objective.name}: {best_pipeline['validation_score']:3f}",
            )
        self._searched = True
        if self.search_iteration_plot is not None:
            if self.verbose and not interactive_plot:
                self.search_iteration_plot = self.plot.search_iteration_plot(
                    interactive_plot=interactive_plot,
                )
                if pio.renderers.default != "browser":
                    self.search_iteration_plot.show()
        return batch_times

    def _find_best_pipeline(self):
        """Finds the best pipeline in the rankings If self._best_pipeline already exists, check to make sure it is different from the current best pipeline before training and thresholding."""
        if len(self.rankings) == 0:
            return
        best_pipeline = self.rankings.iloc[0]
        if not (
            self._best_pipeline
            and self._best_pipeline == self.get_pipeline(best_pipeline["id"])
        ):
            best_pipeline = self.get_pipeline(best_pipeline["id"])
            if self._train_best_pipeline:
                X_train = self.X_train
                y_train = self.y_train
                best_pipeline = self._engine.submit_training_job(
                    self.automl_config,
                    best_pipeline,
                    X_train,
                    y_train,
                ).get_result()[0]

            self._best_pipeline = best_pipeline

    def _num_pipelines(self):
        """Return the number of pipeline evaluations which have been made.

        Returns:
            int: The number of pipeline evaluations made in the search.
        """
        return len(self._results["pipeline_results"])

    def _validate_problem_type(self):
        for obj in self.additional_objectives:
            if not obj.is_defined_for_problem_type(self.problem_type):
                raise ValueError(
                    "Additional objective {} is not compatible with a {} problem.".format(
                        obj.name,
                        self.problem_type.value,
                    ),
                )

    def _get_baseline_pipeline(self):
        """Creates a baseline pipeline instance."""
        classification_component_graph = {
            "Label Encoder": ["Label Encoder", "X", "y"],
            "Baseline Classifier": [
                "Baseline Classifier",
                "Label Encoder.x",
                "Label Encoder.y",
            ],
        }
        if self.problem_type == ProblemTypes.BINARY:
            baseline = BinaryClassificationPipeline(
                component_graph=classification_component_graph,
                custom_name="Mode Baseline Binary Classification Pipeline",
                parameters={"Baseline Classifier": {"strategy": "mode"}},
            )
        elif self.problem_type == ProblemTypes.MULTICLASS:
            baseline = MulticlassClassificationPipeline(
                component_graph=classification_component_graph,
                custom_name="Mode Baseline Multiclass Classification Pipeline",
                parameters={"Baseline Classifier": {"strategy": "mode"}},
            )
        elif self.problem_type == ProblemTypes.REGRESSION:
            baseline = RegressionPipeline(
                component_graph=["Baseline Regressor"],
                custom_name="Mean Baseline Regression Pipeline",
                parameters={"Baseline Regressor": {"strategy": "mean"}},
            )
        else:
            gap = self.problem_configuration["gap"]
            forecast_horizon = self.problem_configuration["forecast_horizon"]
            time_index = self.problem_configuration["time_index"]
            exclude_timeseries_featurizer = (
                "TimeSeriesFeaturizer" in self.exclude_featurizers
            )
            baseline = make_timeseries_baseline_pipeline(
                self.problem_type,
                gap,
                forecast_horizon,
                time_index,
                exclude_timeseries_featurizer,
            )
        return baseline

    def _add_baseline_pipelines(self):
        """Fits a baseline pipeline to the data.

        This is the first pipeline fit during search.
        """
        baseline = self._get_baseline_pipeline()
        self._pre_evaluation_callback(baseline)
        self.logger.info(f"Evaluating Baseline Pipeline: {baseline.name}")
        computation = self._engine.submit_evaluation_job(
            self.automl_config,
            baseline,
            self.X_train,
            self.y_train,
            self.X_holdout,
            self.y_holdout,
        )
        evaluation = computation.get_result()
        data, cached_data, pipeline, job_log = (
            evaluation.get("scores"),
            evaluation.get("cached_data"),
            evaluation.get("pipeline"),
            evaluation.get("logger"),
        )
        self._post_evaluation_callback(pipeline, data, cached_data, job_log)

    @staticmethod
    def _get_mean_cv_scores_for_all_objectives(cv_data, objective_name_to_class):
        scores = defaultdict(int)
        n_folds = len(cv_data)
        for fold_data in cv_data:
            for field, value in fold_data["all_objective_scores"].items():
                # The 'all_objective_scores' field contains scores for all objectives
                # but also fields like "# Training" and "# Testing", so we want to exclude them since
                # they are not scores
                if field in objective_name_to_class:
                    scores[field] += value
        return {
            objective: float(score) / n_folds for objective, score in scores.items()
        }

    def _post_evaluation_callback(
        self,
        pipeline,
        evaluation_results,
        cached_data,
        job_log,
    ):
        job_log.write_to_logger(self.logger)
        training_time = evaluation_results["training_time"]
        cv_data = evaluation_results["cv_data"]
        cv_scores = evaluation_results["cv_scores"]
        is_baseline = pipeline.model_family == ModelFamily.BASELINE
        mean_cv_score = np.nan if len(cv_scores) == 1 else cv_scores.mean()
        if len(cv_scores) == 1 and evaluation_results["holdout_score"] is None:
            validation_score = cv_scores[0]
        elif evaluation_results["holdout_score"] is None:
            validation_score = mean_cv_score
        else:
            validation_score = evaluation_results["holdout_score"]
        cv_sd = cv_scores.std()

        percent_better_than_baseline = {}
        mean_cv_all_objectives = self._get_mean_cv_scores_for_all_objectives(
            cv_data,
            self.objective_name_to_class,
        )
        if is_baseline:
            self._baseline_cv_scores = mean_cv_all_objectives
        for obj_name in mean_cv_all_objectives:
            objective_class = self.objective_name_to_class[obj_name]

            # In the event add_to_rankings is called before search _baseline_cv_scores will be empty so we will return
            # nan for the base score.
            percent_better = objective_class.calculate_percent_difference(
                mean_cv_all_objectives[obj_name],
                self._baseline_cv_scores.get(obj_name, np.nan),
            )
            percent_better_than_baseline[obj_name] = percent_better

        high_variance_cv = self._check_for_high_variance(pipeline, cv_scores)

        pipeline_id = len(self._results["pipeline_results"])
        self._results["pipeline_results"][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline.name,
            "pipeline_class": pipeline.__class__,
            "pipeline_summary": pipeline.summary,
            "parameters": pipeline.parameters,
            "mean_cv_score": mean_cv_score,
            "standard_deviation_cv_score": cv_sd,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data,
            "percent_better_than_baseline_all_objectives": percent_better_than_baseline,
            "percent_better_than_baseline": percent_better_than_baseline[
                self.objective.name
            ],
            "validation_score": validation_score,
        }
        self._pipelines_searched.update({pipeline_id: pipeline.clone()})

        if pipeline.model_family == ModelFamily.ENSEMBLE:
            input_pipeline_ids = [
                self.automl_algorithm._best_pipeline_info[model_family]["id"]
                for model_family in self.automl_algorithm._best_pipeline_info
            ]
            self._results["pipeline_results"][pipeline_id][
                "input_pipeline_ids"
            ] = input_pipeline_ids

        self._results["search_order"].append(pipeline_id)

        if not is_baseline:
            score_to_minimize = (
                -validation_score
                if self.objective.greater_is_better
                else validation_score
            )
            try:
                self.automl_algorithm.add_result(
                    score_to_minimize,
                    pipeline,
                    self._results["pipeline_results"][pipeline_id],
                    cached_data,
                )
            except PipelineNotFoundError:
                pass

        # True when running in a jupyter notebook, else the plot is an instance of plotly.Figure
        if isinstance(self.search_iteration_plot, SearchIterationPlot):
            self.search_iteration_plot.update(self.results, self.objective)

        if self.add_result_callback:
            self.add_result_callback(
                self._results["pipeline_results"][pipeline_id],
                pipeline,
                self,
            )
        return pipeline_id

    def _check_for_high_variance(self, pipeline, cv_scores, threshold=0.5):
        """Checks cross-validation scores and logs a warning if variance is higher than specified threshhold."""
        pipeline_name = pipeline.name

        high_variance_cv = False
        allowed_range = (
            self.objective.expected_range[1] - self.objective.expected_range[0]
        )
        if allowed_range == float("inf"):
            return high_variance_cv
        cv_range = max(cv_scores) - min(cv_scores)
        if cv_range >= threshold * allowed_range:
            self.logger.warning(
                f"\tHigh coefficient of variation (cv >= {threshold}) within cross validation scores.\n\t{pipeline_name} may not perform as estimated on unseen data.",
            )
            high_variance_cv = True
        return high_variance_cv

    def get_pipeline(self, pipeline_id):
        """Given the ID of a pipeline training result, returns an untrained instance of the specified pipeline initialized with the parameters used to train that pipeline during automl search.

        Args:
            pipeline_id (int): Pipeline to retrieve.

        Returns:
            PipelineBase: Untrained pipeline instance associated with the provided ID.

        Raises:
            PipelineNotFoundError: if pipeline_id is not a valid ID.
        """
        pipeline_results = self.results["pipeline_results"].get(pipeline_id)
        if pipeline_results is None:
            raise PipelineNotFoundError("Pipeline not found in automl results")
        pipeline = self._pipelines_searched.get(pipeline_id)
        parameters = pipeline_results.get("parameters")
        if pipeline is None or parameters is None:
            raise PipelineNotFoundError(
                "Pipeline class or parameters not found in automl results",
            )
        new_pipeline = pipeline.new(parameters, random_seed=self.random_seed)
        if is_binary(self.problem_type):
            new_pipeline.threshold = None
        return new_pipeline

    def describe_pipeline(self, pipeline_id, return_dict=False):
        """Describe a pipeline.

        Args:
            pipeline_id (int): pipeline to describe
            return_dict (bool): If True, return dictionary of information
                about pipeline. Defaults to False.

        Returns:
            Description of specified pipeline. Includes information such as
            type of pipeline components, problem, training time, cross validation, etc.

        Raises:
            PipelineNotFoundError: If pipeline_id is not a valid ID.
        """
        logger = get_logger(f"{__name__}.describe_pipeline")
        if pipeline_id not in self._results["pipeline_results"]:
            raise PipelineNotFoundError("Pipeline not found")

        pipeline = self.get_pipeline(pipeline_id)
        pipeline_results = self._results["pipeline_results"][pipeline_id]

        pipeline.describe()

        if pipeline.model_family == ModelFamily.ENSEMBLE:
            logger.info(
                "Input for ensembler are pipelines with IDs: "
                + str(pipeline_results["input_pipeline_ids"]),
            )

        log_subtitle(logger, "Training")
        logger.info("Training for {} problems.".format(pipeline.problem_type))

        if (
            self.optimize_thresholds
            and self.objective.is_defined_for_problem_type(ProblemTypes.BINARY)
            and self.objective.can_optimize_threshold
        ):
            logger.info(
                "Objective to optimize binary classification pipeline thresholds for: {}".format(
                    self.objective,
                ),
            )

        logger.info(
            "Total training time (including CV): %.1f seconds"
            % pipeline_results["training_time"],
        )
        log_subtitle(logger, "Cross Validation", underline="-")

        all_objective_scores = [
            fold["all_objective_scores"] for fold in pipeline_results["cv_data"]
        ]
        all_objective_scores = pd.DataFrame(all_objective_scores)

        for c in all_objective_scores:
            if c in ["# Training", "# Validation"]:
                all_objective_scores[c] = all_objective_scores[c].map(
                    lambda x: "{:2,.0f}".format(x) if not pd.isna(x) else np.nan,
                )
                continue

            mean = all_objective_scores[c].mean(axis=0)
            std = all_objective_scores[c].std(axis=0)
            all_objective_scores.loc["mean", c] = mean
            all_objective_scores.loc["std", c] = std
            all_objective_scores.loc["coef of var", c] = (
                std / mean if abs(mean) > 0 else np.inf
            )

        all_objective_scores = all_objective_scores.fillna("-")

        with pd.option_context(
            "display.float_format",
            "{:.3f}".format,
            "expand_frame_repr",
            False,
        ):
            logger.info(all_objective_scores)

        if return_dict:
            return pipeline_results

    def add_to_rankings(self, pipeline):
        """Fits and evaluates a given pipeline then adds the results to the automl rankings with the requirement that automl search has been run.

        Args:
            pipeline (PipelineBase): pipeline to train and evaluate.
        """
        pipeline_rows = self.full_rankings[
            self.full_rankings["pipeline_name"] == pipeline.name
        ]
        for parameter in pipeline_rows["parameters"]:
            if pipeline.parameters == parameter:
                return

        computation = self._engine.submit_evaluation_job(
            self.automl_config,
            pipeline,
            self.X_train,
            self.y_train,
            self.X_holdout,
            self.y_holdout,
        )
        evaluation = computation.get_result()
        data, cached_data, pipeline, job_log = (
            evaluation.get("scores"),
            evaluation.get("cached_data"),
            evaluation.get("pipeline"),
            evaluation.get("logger"),
        )
        self._post_evaluation_callback(pipeline, data, cached_data, job_log)
        self._find_best_pipeline()

    @property
    def results(self):
        """Class that allows access to a copy of the results from `automl_search`.

        Returns:
            dict: Dictionary containing `pipeline_results`, a dict with results from each pipeline,
                 and `search_order`, a list describing the order the pipelines were searched.
        """
        return copy.deepcopy(self._results)

    @property
    def rankings(self):
        """Returns a pandas.DataFrame with scoring results from the highest-scoring set of parameters used with each pipeline."""
        return self.full_rankings.drop_duplicates(subset="pipeline_name", keep="first")

    @property
    def full_rankings(self):
        """Returns a pandas.DataFrame with scoring results from all pipelines searched."""
        ascending = True
        if self.objective.greater_is_better:
            ascending = False
        pipeline_results_cols = [
            "id",
            "pipeline_name",
            "validation_score",
            "mean_cv_score",
            "standard_deviation_cv_score",
            "percent_better_than_baseline",
            "high_variance_cv",
            "parameters",
        ]

        if not self._results["pipeline_results"]:
            full_rankings_cols = (
                pipeline_results_cols[0:2]
                + ["search_order"]
                + pipeline_results_cols[2:]
            )  # place search_order after pipeline_name

            return pd.DataFrame(columns=full_rankings_cols)

        rankings_df = pd.DataFrame(self._results["pipeline_results"].values())
        rankings_df = rankings_df[pipeline_results_cols]
        rankings_df.insert(
            2,
            "search_order",
            pd.Series(self._results["search_order"]),
        )  # place search_order after pipeline_name
        rankings_df.sort_values("validation_score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)
        return rankings_df

    @property
    def best_pipeline(self):
        """Returns a trained instance of the best pipeline and parameters found during automl search. If `train_best_pipeline` is set to False, returns an untrained pipeline instance.

        Returns:
            PipelineBase: A trained instance of the best pipeline and parameters found during automl search. If `train_best_pipeline` is set to False, returns an untrained pipeline instance.

        Raises:
            PipelineNotFoundError: If this is called before .search() is called.
        """
        if not self._best_pipeline:
            raise PipelineNotFoundError(
                "automl search must be run before selecting `best_pipeline`.",
            )

        return self._best_pipeline

    def save(
        self,
        file_path,
        pickle_type="cloudpickle",
        pickle_protocol=cloudpickle.DEFAULT_PROTOCOL,
    ):
        """Saves AutoML object at file path.

        Args:
            file_path (str): Location to save file.
            pickle_type ({"pickle", "cloudpickle"}): The pickling library to use.
            pickle_protocol (int): The pickle data stream format.

        Raises:
            ValueError: If pickle_type is not "pickle" or "cloudpickle".
        """
        if pickle_type == "cloudpickle":
            pkl_lib = cloudpickle
        elif pickle_type == "pickle":
            pkl_lib = pickle
        else:
            raise ValueError(
                f"`pickle_type` must be either 'pickle' or 'cloudpickle'. Received {pickle_type}",
            )

        with open(file_path, "wb") as f:
            pkl_lib.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(
        file_path,
        pickle_type="cloudpickle",
    ):
        """Loads AutoML object at file path.

        Args:
            file_path (str): Location to find file to load
            pickle_type ({"pickle", "cloudpickle"}): The pickling library to use. Currently not used since the standard pickle library can handle cloudpickles.

        Returns:
            AutoSearchBase object
        """
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def train_pipelines(self, pipelines):
        """Train a list of pipelines on the training data.

        This can be helpful for training pipelines once the search is complete.

        Args:
            pipelines (list[PipelineBase]): List of pipelines to train.

        Returns:
            Dict[str, PipelineBase]: Dictionary keyed by pipeline name that maps to the fitted pipeline.
            Note that the any pipelines that error out during training will not be included in the dictionary
            but the exception and stacktrace will be displayed in the log.
        """
        check_all_pipeline_names_unique(pipelines)
        fitted_pipelines = {}
        computations = []
        X_train = self.X_train
        y_train = self.y_train

        for pipeline in pipelines:
            computations.append(
                self._engine.submit_training_job(
                    self.automl_config,
                    pipeline,
                    X_train,
                    y_train,
                ),
            )

        while computations:
            computation = computations.pop(0)
            if computation.done():
                try:
                    fitted_pipeline = computation.get_result()[0]
                    fitted_pipelines[fitted_pipeline.name] = fitted_pipeline
                except Exception as e:
                    self.logger.error(f"Train error for {pipeline.name}: {str(e)}")
                    tb = traceback.format_tb(sys.exc_info()[2])
                    self.logger.error("Traceback:")
                    self.logger.error("\n".join(tb))
            else:
                computations.append(computation)

        return fitted_pipelines

    def score_pipelines(self, pipelines, X_holdout, y_holdout, objectives):
        """Score a list of pipelines on the given holdout data.

        Args:
            pipelines (list[PipelineBase]): List of pipelines to train.
            X_holdout (pd.DataFrame): Holdout features.
            y_holdout (pd.Series): Holdout targets for scoring.
            objectives (list[str], list[ObjectiveBase]): Objectives used for scoring.

        Returns:
            dict[str, Dict[str, float]]: Dictionary keyed by pipeline name that maps to a dictionary of scores.
            Note that the any pipelines that error out during scoring will not be included in the dictionary
            but the exception and stacktrace will be displayed in the log.
        """
        X_holdout, y_holdout = infer_feature_types(X_holdout), infer_feature_types(
            y_holdout,
        )
        check_all_pipeline_names_unique(pipelines)
        scores = {}
        objectives = [get_objective(o, return_instance=True) for o in objectives]

        computations = []
        for pipeline in pipelines:
            X_train, y_train = None, None
            if is_time_series(self.problem_type):
                X_train, y_train = self.X_train, self.y_train
            computations.append(
                self._engine.submit_scoring_job(
                    self.automl_config,
                    pipeline,
                    X_holdout,
                    y_holdout,
                    objectives,
                    X_train=X_train,
                    y_train=y_train,
                ),
            )

        while computations:
            computation = computations.pop(0)
            if computation.done():
                pipeline_name = computation.meta_data["pipeline_name"]
                try:
                    scores[pipeline_name] = computation.get_result()
                except Exception as e:
                    self.logger.error(f"Score error for {pipeline_name}: {str(e)}")
                    if isinstance(e, PipelineScoreError):
                        nan_scores = {objective: np.nan for objective in e.exceptions}
                        scores[pipeline_name] = {**nan_scores, **e.scored_successfully}
                    else:
                        # Traceback already included in the PipelineScoreError so we only
                        # need to include it for all other errors
                        tb = traceback.format_tb(sys.exc_info()[2])
                        self.logger.error("Traceback:")
                        self.logger.error("\n".join(tb))
                        scores[pipeline_name] = {
                            objective.name: np.nan for objective in objectives
                        }
            else:
                computations.append(computation)
        return scores

    @property
    def plot(self):
        """Return an instance of the plot with the latest scores."""
        return PipelineSearchPlots(self.results, self.objective)

    @property
    def _sleep_time(self):
        return self._SLEEP_TIME

    def get_ensembler_input_pipelines(self, ensemble_pipeline_id):
        """Returns a list of input pipeline IDs given an ensembler pipeline ID.

        Args:
            ensemble_pipeline_id (id): Ensemble pipeline ID to get input pipeline IDs from.

        Returns:
            list[int]: A list of ensemble input pipeline IDs.

        Raises:
            ValueError: If `ensemble_pipeline_id` does not correspond to a valid ensemble pipeline ID.
        """
        pipeline_results = self._results["pipeline_results"]
        if (
            ensemble_pipeline_id not in pipeline_results
            or "input_pipeline_ids" not in pipeline_results[ensemble_pipeline_id]
        ):
            raise ValueError(
                f"Pipeline ID {ensemble_pipeline_id} is not a valid ensemble pipeline",
            )

        return self._results["pipeline_results"][ensemble_pipeline_id][
            "input_pipeline_ids"
        ]
