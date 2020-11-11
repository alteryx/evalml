import copy
import sys
import time
import traceback
from collections import OrderedDict, defaultdict

import cloudpickle
import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    train_test_split
)

from .pipeline_search_plots import PipelineSearchPlots

from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.callbacks import log_error_callback
from evalml.automl.data_splitters import TrainingValidationSplit
from evalml.automl.utils import get_default_primary_search_objective
from evalml.data_checks import (
    AutoMLDataChecks,
    DataChecks,
    DefaultDataChecks,
    EmptyDataChecks,
    HighVarianceCVDataCheck
)
from evalml.data_checks.data_check_message_type import DataCheckMessageType
from evalml.exceptions import (
    AutoMLSearchException,
    PipelineNotFoundError,
    PipelineScoreError
)
from evalml.model_family import ModelFamily
from evalml.objectives import (
    get_all_objective_names,
    get_core_objectives,
    get_non_core_objectives,
    get_objective
)
from evalml.pipelines import (
    BinaryClassificationPipeline,
    MeanBaselineRegressionPipeline,
    ModeBaselineBinaryPipeline,
    ModeBaselineMulticlassPipeline
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.tuners import SKOptTuner
from evalml.utils import convert_to_seconds, get_random_state
from evalml.utils.gen_utils import (
    _convert_to_woodwork_structure,
    _convert_woodwork_types_wrapper
)
from evalml.utils.logger import (
    get_logger,
    log_subtitle,
    log_title,
    time_elapsed,
    update_pipeline
)

logger = get_logger(__file__)


class AutoMLSearch:
    """Automated Pipeline search."""
    _MAX_NAME_LEN = 40
    _LARGE_DATA_ROW_THRESHOLD = int(1e5)
    _LARGE_DATA_PERCENT_VALIDATION = 0.75

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    def __init__(self,
                 problem_type=None,
                 objective='auto',
                 max_iterations=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 data_split=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 error_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 tuner_class=None,
                 verbose=True,
                 optimize_thresholds=False,
                 ensembling=False,
                 max_batches=None,
                 _pipelines_per_batch=5):
        """Automated pipeline search

        Arguments:
            problem_type (str or ProblemTypes): Choice of 'regression', 'binary', or 'multiclass', depending on the desired problem type.

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

            allowed_pipelines (list(class)): A list of PipelineBase subclasses indicating the pipelines allowed in the search.
                The default of None indicates all pipelines for this problem type are allowed. Setting this field will cause
                allowed_model_families to be ignored.

            allowed_model_families (list(str, ModelFamily)): The model families to search. The default of None searches over all
                model families. Run evalml.pipelines.components.utils.allowed_model_families("binary") to see options. Change `binary`
                to `multiclass` or `regression` depending on the problem type. Note that if allowed_pipelines is provided,
                this parameter will be ignored.

            data_split (sklearn.model_selection.BaseCrossValidator): Data splitting method to use. Defaults to StratifiedKFold.

            tuner_class: The tuner class to use. Defaults to SKOptTuner.

            start_iteration_callback (callable): Function called before each pipeline training iteration.
                Callback function takes three positional parameters: The pipeline class, the pipeline parameters, and the AutoMLSearch object.

            add_result_callback (callable): Function called after each pipeline training iteration.
                Callback function takes three positional parameters:: A dictionary containing the training results for the new pipeline, an untrained_pipeline containing the parameters used during training, and the AutoMLSearch object.

            error_callback (callable): Function called when `search()` errors and raises an Exception.
                Callback function takes three positional parameters: the Exception raised, the traceback, and the AutoMLSearch object.
                Must also accepts kwargs, so AutoMLSearch is able to pass along other appropriate parameters by default.
                Defaults to None, which will call `log_error_callback`.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            verbose (boolean): If True, turn verbosity on. Defaults to True.

            ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over.
                If the number of unique pipelines to search over per batch is one, ensembling will not run. Defaults to False.

            max_batches (int): The maximum number of batches of pipelines to search. Parameters max_time, and
                max_iterations have precedence over stopping the search.

            _pipelines_per_batch (int): The number of pipelines to train for every batch after the first one.
                The first batch will train a baseline pipline + one of each pipeline family allowed in the search.
        """
        try:
            self.problem_type = handle_problem_types(problem_type)
        except ValueError:
            raise ValueError('choose one of (binary, multiclass, regression) as problem_type')

        self.tuner_class = tuner_class or SKOptTuner
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.error_callback = error_callback or log_error_callback
        self.data_split = data_split
        self.verbose = verbose
        self.optimize_thresholds = optimize_thresholds
        self.ensembling = ensembling
        if objective == 'auto':
            objective = get_default_primary_search_objective(self.problem_type.value)
        objective = get_objective(objective, return_instance=False)
        self.objective = self._validate_objective(objective)
        if self.data_split is not None and not issubclass(self.data_split.__class__, BaseCrossValidator):
            raise ValueError("Not a valid data splitter")
        if not objective.is_defined_for_problem_type(self.problem_type):
            raise ValueError("Given objective {} is not compatible with a {} problem.".format(self.objective.name, self.problem_type.value))
        if additional_objectives is None:
            additional_objectives = get_core_objectives(self.problem_type)
            # if our main objective is part of default set of objectives for problem_type, remove it
            existing_main_objective = next((obj for obj in additional_objectives if obj.name == self.objective.name), None)
            if existing_main_objective is not None:
                additional_objectives.remove(existing_main_objective)
        else:
            additional_objectives = [get_objective(o) for o in additional_objectives]
        additional_objectives = [self._validate_objective(obj) for obj in additional_objectives]
        self.additional_objectives = additional_objectives

        if max_time is None or isinstance(max_time, (int, float)):
            self.max_time = max_time
        elif isinstance(max_time, str):
            self.max_time = convert_to_seconds(max_time)
        else:
            raise TypeError("max_time must be a float, int, or string. Received a {}.".format(type(max_time)))

        if max_batches is not None and max_batches <= 0:
            raise ValueError(f"Parameter max batches must be None or non-negative. Received {max_batches}.")
        self.max_batches = max_batches
        self._pipelines_per_batch = _pipelines_per_batch

        self.max_iterations = max_iterations
        if not self.max_iterations and not self.max_time and not self.max_batches:
            self.max_iterations = 5
            logger.info("Using default limit of max_iterations=5.\n")

        if patience and (not isinstance(patience, int) or patience < 0):
            raise ValueError("patience value must be a positive integer. Received {} instead".format(patience))

        if tolerance and (tolerance > 1.0 or tolerance < 0.0):
            raise ValueError("tolerance value must be a float between 0.0 and 1.0 inclusive. Received {} instead".format(tolerance))

        self.patience = patience
        self.tolerance = tolerance or 0.0
        self._results = {
            'pipeline_results': {},
            'search_order': [],
            'errors': []
        }
        self.random_state = get_random_state(random_state)
        self.n_jobs = n_jobs

        self.plot = None
        try:
            self.plot = PipelineSearchPlots(self)
        except ImportError:
            logger.warning("Unable to import plotly; skipping pipeline search plotting\n")

        self._data_check_results = None

        self.allowed_pipelines = allowed_pipelines
        self.allowed_model_families = allowed_model_families
        self._automl_algorithm = None
        self._start = None
        self._baseline_cv_scores = {}

        self._validate_problem_type()

    def _validate_objective(self, objective):
        non_core_objectives = get_non_core_objectives()
        if isinstance(objective, type):
            if objective in non_core_objectives:
                raise ValueError(f"{objective.name.lower()} is not allowed in AutoML! "
                                 "Use evalml.objectives.utils.get_core_objective_names() "
                                 "to get all objective names allowed in automl.")
            return objective()
        return objective

    @property
    def data_check_results(self):
        """If there are data checks, return any error messages that are found"""
        return self._data_check_results

    def __str__(self):
        def _print_list(obj_list):
            lines = sorted(['\t{}'.format(o.name) for o in obj_list])
            return '\n'.join(lines)

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
            f"Allowed Pipelines: \n{_print_list(self.allowed_pipelines or [])}\n"
            f"Patience: {self.patience}\n"
            f"Tolerance: {self.tolerance}\n"
            f"Data Splitting: {self.data_split}\n"
            f"Tuner: {self.tuner_class.__name__}\n"
            f"Start Iteration Callback: {_get_funct_name(self.start_iteration_callback)}\n"
            f"Add Result Callback: {_get_funct_name(self.add_result_callback)}\n"
            f"Additional Objectives: {_print_list(self.additional_objectives or [])}\n"
            f"Random State: {self.random_state}\n"
            f"n_jobs: {self.n_jobs}\n"
            f"Verbose: {self.verbose}\n"
            f"Optimize Thresholds: {self.optimize_thresholds}\n"
        )

        rankings_desc = ""
        if not self.rankings.empty:
            rankings_str = self.rankings.drop(['parameters'], axis='columns').to_string()
            rankings_desc = f"\nSearch Results: \n{'='*20}\n{rankings_str}"

        return search_desc + rankings_desc

    def _validate_data_checks(self, data_checks):
        """Validate data_checks parameter.

        Arguments:
            data_checks (DataChecks, list(Datacheck), str, None): Input to validate. If not of the right type,
                raise an exception.

        Returns:
            An instance of DataChecks used to perform checks before search.
        """
        if isinstance(data_checks, DataChecks):
            return data_checks
        elif isinstance(data_checks, list):
            return AutoMLDataChecks(data_checks)
        elif isinstance(data_checks, str):
            if data_checks == "auto":
                return DefaultDataChecks(problem_type=self.problem_type)
            elif data_checks == "disabled":
                return EmptyDataChecks()
            else:
                raise ValueError("If data_checks is a string, it must be either 'auto' or 'disabled'. "
                                 f"Received '{data_checks}'.")
        elif data_checks is None:
            return EmptyDataChecks()
        else:
            return DataChecks(data_checks)

    def _handle_keyboard_interrupt(self, pipeline, current_batch_pipelines):
        """Presents a prompt to the user asking if they want to stop the search.

        Arguments:
            pipeline (PipelineBase): Current pipeline in the search.
            current_batch_pipelines (list): Other pipelines in the batch.

        Returns:
            list: Next pipelines to search in the batch. If the user decides to stop the search,
                an empty list will be returned.
        """
        leading_char = "\n"
        start_of_loop = time.time()
        while True:
            choice = input(leading_char + "Do you really want to exit search (y/n)? ").strip().lower()
            if choice == "y":
                logger.info("Exiting AutoMLSearch.")
                return []
            elif choice == "n":
                # So that the time in this loop does not count towards the time budget (if set)
                time_in_loop = time.time() - start_of_loop
                self._start += time_in_loop
                return [pipeline] + current_batch_pipelines
            else:
                leading_char = ""

    def _set_data_split(self, X):
        """Sets the data split method for AutoMLSearch

        Arguments:
            X (DataFrame): Input dataframe to split
        """
        if self.problem_type == ProblemTypes.REGRESSION:
            default_data_split = KFold(n_splits=3, random_state=self.random_state, shuffle=True)
        elif self.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
            default_data_split = StratifiedKFold(n_splits=3, random_state=self.random_state, shuffle=True)

        if X.shape[0] > self._LARGE_DATA_ROW_THRESHOLD:
            default_data_split = TrainingValidationSplit(test_size=self._LARGE_DATA_PERCENT_VALIDATION, shuffle=True)

        self.data_split = self.data_split or default_data_split

    def search(self, X, y, data_checks="auto", show_iteration_plot=True):
        """Find the best pipeline for the data set.

        Arguments:
            X (pd.DataFrame, ww.DataTable): The input training data of shape [n_samples, n_features]

            y (pd.Series, ww.DataColumn): The target training data of length [n_samples]

            show_iteration_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.

            data_checks (DataChecks, list(Datacheck), str, None): A collection of data checks to run before
                automl search. If data checks produce any errors, an exception will be thrown before the
                search begins. If "disabled" or None, no data checks will be done.
                If set to "auto", DefaultDataChecks will be done. Default value is set to "auto".

        Returns:
            self
        """
        # don't show iteration plot outside of a jupyter notebook
        if show_iteration_plot:
            try:
                get_ipython
            except NameError:
                show_iteration_plot = False

        # make everything ww objects
        if not isinstance(X, ww.DataTable):
            logger.warning("`X` passed was not a DataTable. EvalML will try to convert the input as a Woodwork DataTable and types will be inferred. To control this behavior, please pass in a Woodwork DataTable instead.")
            X = _convert_to_woodwork_structure(X)

        text_column_vals = X.select('natural_language')
        text_columns = list(text_column_vals.to_dataframe().columns)
        if len(text_columns) == 0:
            text_columns = None

        if not isinstance(y, ww.DataColumn):
            logger.warning("`y` passed was not a DataColumn. EvalML will try to convert the input as a Woodwork DataTable and types will be inferred. To control this behavior, please pass in a Woodwork DataTable instead.")
            y = _convert_to_woodwork_structure(y)

        X = _convert_woodwork_types_wrapper(X.to_dataframe())
        y = _convert_woodwork_types_wrapper(y.to_dataframe())

        self._set_data_split(X)

        data_checks = self._validate_data_checks(data_checks)
        data_check_results = data_checks.validate(X, y)
        if data_check_results:
            self._data_check_results = data_check_results
            for message in self._data_check_results:
                if message.message_type == DataCheckMessageType.WARNING:
                    logger.warning(message)
                elif message.message_type == DataCheckMessageType.ERROR:
                    logger.error(message)
            if any([message.message_type == DataCheckMessageType.ERROR for message in self._data_check_results]):
                raise ValueError("Data checks raised some warnings and/or errors. Please see `self.data_check_results` for more information or pass data_checks='disabled' to search() to disable data checking.")

        if self.allowed_pipelines is None:
            logger.info("Generating pipelines to search over...")
            allowed_estimators = get_estimators(self.problem_type, self.allowed_model_families)
            logger.debug(f"allowed_estimators set to {[estimator.name for estimator in allowed_estimators]}")
            self.allowed_pipelines = [make_pipeline(X, y, estimator, self.problem_type, text_columns=text_columns) for estimator in allowed_estimators]

        if self.allowed_pipelines == []:
            raise ValueError("No allowed pipelines to search")

        run_ensembling = self.ensembling
        if run_ensembling and len(self.allowed_pipelines) == 1:
            logger.warning("Ensembling was set to True, but the number of unique pipelines is one, so ensembling will not run.")
            run_ensembling = False

        if self.max_batches and self.max_iterations is None:
            if run_ensembling:
                ensemble_nth_batch = len(self.allowed_pipelines) + 1
                num_ensemble_batches = (self.max_batches - 1) // ensemble_nth_batch
                self.max_iterations = (1 + len(self.allowed_pipelines) +
                                       self._pipelines_per_batch * (self.max_batches - 1 - num_ensemble_batches) +
                                       num_ensemble_batches)
            else:
                self.max_iterations = 1 + len(self.allowed_pipelines) + (self._pipelines_per_batch * (self.max_batches - 1))
        self.allowed_model_families = list(set([p.model_family for p in (self.allowed_pipelines)]))

        logger.debug(f"allowed_pipelines set to {[pipeline.name for pipeline in self.allowed_pipelines]}")
        logger.debug(f"allowed_model_families set to {self.allowed_model_families}")

        self._automl_algorithm = IterativeAlgorithm(
            max_iterations=self.max_iterations,
            allowed_pipelines=self.allowed_pipelines,
            tuner_class=self.tuner_class,
            text_columns=text_columns,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            number_features=X.shape[1],
            pipelines_per_batch=self._pipelines_per_batch,
            ensembling=run_ensembling
        )

        log_title(logger, "Beginning pipeline search")
        logger.info("Optimizing for %s. " % self.objective.name)
        logger.info("{} score is better.\n".format('Greater' if self.objective.greater_is_better else 'Lower'))

        if self.max_batches is not None:
            logger.info(f"Searching up to {self.max_batches} batches for a total of {self.max_iterations} pipelines. ")
        elif self.max_iterations is not None:
            logger.info("Searching up to %s pipelines. " % self.max_iterations)
        if self.max_time is not None:
            logger.info("Will stop searching for new pipelines after %d seconds.\n" % self.max_time)
        logger.info("Allowed model families: %s\n" % ", ".join([model.value for model in self.allowed_model_families]))
        search_iteration_plot = None
        if self.plot:
            search_iteration_plot = self.plot.search_iteration_plot(interactive_plot=show_iteration_plot)

        self._start = time.time()

        should_terminate = self._add_baseline_pipelines(X, y)
        if should_terminate:
            return

        current_batch_pipelines = []
        current_batch_pipeline_scores = []
        while self._check_stopping_condition(self._start):
            try:
                if len(current_batch_pipelines) == 0:
                    try:
                        if current_batch_pipeline_scores and np.isnan(np.array(current_batch_pipeline_scores, dtype=float)).all():
                            raise AutoMLSearchException(f"All pipelines in the current AutoML batch produced a score of np.nan on the primary objective {self.objective}.")
                        current_batch_pipelines = self._automl_algorithm.next_batch()
                        current_batch_pipeline_scores = []
                    except StopIteration:
                        logger.info('AutoML Algorithm out of recommendations, ending')
                        break
                pipeline = current_batch_pipelines.pop(0)
                parameters = pipeline.parameters
                logger.debug('Evaluating pipeline {}'.format(pipeline.name))
                logger.debug('Pipeline parameters: {}'.format(parameters))

                if self.start_iteration_callback:
                    self.start_iteration_callback(pipeline.__class__, parameters, self)
                desc = f"{pipeline.name}"
                if len(desc) > self._MAX_NAME_LEN:
                    desc = desc[:self._MAX_NAME_LEN - 3] + "..."
                desc = desc.ljust(self._MAX_NAME_LEN)

                if self.max_batches:
                    update_pipeline(logger, desc, len(self._results['pipeline_results']) + 1, self.max_iterations, self._start, self._automl_algorithm.batch_number)
                else:
                    update_pipeline(logger, desc, len(self._results['pipeline_results']) + 1, self.max_iterations, self._start)

                evaluation_results = self._evaluate(pipeline, X, y)
                score = evaluation_results['cv_score_mean']
                score_to_minimize = -score if self.objective.greater_is_better else score
                current_batch_pipeline_scores.append(score_to_minimize)
                self._automl_algorithm.add_result(score_to_minimize, pipeline)

                if search_iteration_plot:
                    search_iteration_plot.update()

            except KeyboardInterrupt:
                current_batch_pipelines = self._handle_keyboard_interrupt(pipeline, current_batch_pipelines)
                if not current_batch_pipelines:
                    return

        elapsed_time = time_elapsed(self._start)
        desc = f"\nSearch finished after {elapsed_time}"
        desc = desc.ljust(self._MAX_NAME_LEN)
        logger.info(desc)

        best_pipeline = self.rankings.iloc[0]
        best_pipeline_name = best_pipeline["pipeline_name"]
        logger.info(f"Best pipeline: {best_pipeline_name}")
        logger.info(f"Best pipeline {self.objective.name}: {best_pipeline['score']:3f}")

    def _check_stopping_condition(self, start):
        should_continue = True
        num_pipelines = len(self._results['pipeline_results'])

        # check max_time and max_iterations
        elapsed = time.time() - start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_iterations and num_pipelines >= self.max_iterations:
            return False

        # check for early stopping
        if self.patience is None:
            return True

        first_id = self._results['search_order'][0]
        best_score = self._results['pipeline_results'][first_id]['score']
        num_without_improvement = 0
        for id in self._results['search_order'][1:]:
            curr_score = self._results['pipeline_results'][id]['score']
            significant_change = abs((curr_score - best_score) / best_score) > self.tolerance
            score_improved = curr_score > best_score if self.objective.greater_is_better else curr_score < best_score
            if score_improved and significant_change:
                best_score = curr_score
                num_without_improvement = 0
            else:
                num_without_improvement += 1
            if num_without_improvement >= self.patience:
                logger.info("\n\n{} iterations without improvement. Stopping search early...".format(self.patience))
                return False
        return should_continue

    def _validate_problem_type(self):
        for obj in self.additional_objectives:
            if not obj.is_defined_for_problem_type(self.problem_type):
                raise ValueError("Additional objective {} is not compatible with a {} problem.".format(obj.name, self.problem_type.value))

        for pipeline in self.allowed_pipelines or []:
            if pipeline.problem_type != self.problem_type:
                raise ValueError("Given pipeline {} is not compatible with problem_type {}.".format(pipeline.name, self.problem_type.value))

    def _add_baseline_pipelines(self, X, y):
        """Fits a baseline pipeline to the data.

        This is the first pipeline fit during search.

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training data of length [n_samples]

        Returns:
            bool - If the user ends the search early, will return True and searching will immediately finish. Else,
                will return False and more pipelines will be searched.
        """

        if self.problem_type == ProblemTypes.BINARY:
            baseline = ModeBaselineBinaryPipeline(parameters={})
        elif self.problem_type == ProblemTypes.MULTICLASS:
            baseline = ModeBaselineMulticlassPipeline(parameters={})
        elif self.problem_type == ProblemTypes.REGRESSION:
            baseline = MeanBaselineRegressionPipeline(parameters={})

        pipelines = [baseline]
        # Using a while loop so that we can retry the pipeline after the user hits ctr-c
        # but decides to not stop the search.
        while pipelines:
            try:
                if self.start_iteration_callback:
                    self.start_iteration_callback(baseline.__class__, baseline.parameters, self)
                baseline = pipelines.pop()
                desc = f"{baseline.name}"
                if len(desc) > self._MAX_NAME_LEN:
                    desc = desc[:self._MAX_NAME_LEN - 3] + "..."
                desc = desc.ljust(self._MAX_NAME_LEN)

                update_pipeline(logger, desc, len(self._results['pipeline_results']) + 1, self.max_iterations,
                                self._start, 1)

                baseline_results = self._compute_cv_scores(baseline, X, y)
                self._baseline_cv_scores = self._get_mean_cv_scores_for_all_objectives(baseline_results["cv_data"])
                self._add_result(trained_pipeline=baseline,
                                 parameters=baseline.parameters,
                                 training_time=baseline_results['training_time'],
                                 cv_data=baseline_results['cv_data'],
                                 cv_scores=baseline_results['cv_scores'])
            except KeyboardInterrupt:
                pipelines = self._handle_keyboard_interrupt(baseline, pipelines)
                if not pipelines:
                    return True

        return False

    @staticmethod
    def _get_mean_cv_scores_for_all_objectives(cv_data):
        scores = defaultdict(int)
        objective_names = set([name.lower() for name in get_all_objective_names()])
        n_folds = len(cv_data)
        for fold_data in cv_data:
            for field, value in fold_data['all_objective_scores'].items():
                if field.lower() in objective_names:
                    scores[field] += value
        return {objective_name: float(score) / n_folds for objective_name, score in scores.items()}

    def _compute_cv_scores(self, pipeline, X, y):
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        for i, (train, test) in enumerate(self.data_split.split(X, y)):
            if pipeline.model_family == ModelFamily.ENSEMBLE and i > 0:
                # Stacked ensembles do CV internally, so we do not run CV here for performance reasons.
                logger.debug(f"Skipping fold {i} because CV for stacked ensembles is not supported.")
                break
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            if self.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
                diff_train = set(np.setdiff1d(y, y_train))
                diff_test = set(np.setdiff1d(y, y_test))
                diff_string = f"Missing target values in the training set after data split: {diff_train}. " if diff_train else ""
                diff_string += f"Missing target values in the test set after data split: {diff_test}." if diff_test else ""
                if diff_string:
                    raise Exception(diff_string)
            objectives_to_score = [self.objective] + self.additional_objectives
            cv_pipeline = None
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None
                if self.optimize_thresholds and self.objective.is_defined_for_problem_type(ProblemTypes.BINARY) and self.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state)
                cv_pipeline = pipeline.clone(pipeline.random_state)
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if self.objective.is_defined_for_problem_type(ProblemTypes.BINARY):
                    cv_pipeline.threshold = 0.5
                    if self.optimize_thresholds and self.objective.can_optimize_threshold:
                        logger.debug(f"\t\t\tFold {i}: Optimizing threshold for {self.objective.name}")
                        y_predict_proba = cv_pipeline.predict_proba(X_threshold_tuning)
                        if isinstance(y_predict_proba, pd.DataFrame):
                            y_predict_proba = y_predict_proba.iloc[:, 1]
                        else:
                            y_predict_proba = y_predict_proba[:, 1]
                        cv_pipeline.threshold = self.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                        logger.debug(f"\t\t\tFold {i}: Optimal threshold found ({cv_pipeline.threshold:.3f})")
                logger.debug(f"\t\t\tFold {i}: Scoring trained pipeline")
                scores = cv_pipeline.score(X_test, y_test, objectives=objectives_to_score)
                logger.debug(f"\t\t\tFold {i}: {self.objective.name} score: {scores[self.objective.name]:.3f}")
                score = scores[self.objective.name]
            except Exception as e:
                if self.error_callback is not None:
                    self.error_callback(exception=e, traceback=traceback.format_tb(sys.exc_info()[2]), automl=self,
                                        fold_num=i, pipeline=pipeline)
                if isinstance(e, PipelineScoreError):
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores = {**nan_scores, **e.scored_successfully}
                    scores = OrderedDict({o.name: scores[o.name] for o in [self.objective] + self.additional_objectives})
                    score = scores[self.objective.name]
                else:
                    score = np.nan
                    scores = OrderedDict(zip([n.name for n in self.additional_objectives], [np.nan] * len(self.additional_objectives)))

            ordered_scores = OrderedDict()
            ordered_scores.update({self.objective.name: score})
            ordered_scores.update(scores)
            ordered_scores.update({"# Training": len(y_train)})
            ordered_scores.update({"# Testing": len(y_test)})

            evaluation_entry = {"all_objective_scores": ordered_scores, "score": score, 'binary_classification_threshold': None}
            if isinstance(cv_pipeline, BinaryClassificationPipeline) and cv_pipeline.threshold is not None:
                evaluation_entry['binary_classification_threshold'] = cv_pipeline.threshold
            cv_data.append(evaluation_entry)
        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        cv_score_mean = cv_scores.mean()
        logger.info(f"\tFinished cross validation - mean {self.objective.name}: {cv_score_mean:.3f}")
        return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_score_mean}

    def _add_result(self, trained_pipeline, parameters, training_time, cv_data, cv_scores):
        cv_score = cv_scores.mean()

        percent_better_than_baseline = {}
        mean_cv_all_objectives = self._get_mean_cv_scores_for_all_objectives(cv_data)
        for obj_name in mean_cv_all_objectives:
            objective_class = get_objective(obj_name)
            # In the event add_to_rankings is called before search _baseline_cv_scores will be empty so we will return
            # nan for the base score.
            percent_better = objective_class.calculate_percent_difference(mean_cv_all_objectives[obj_name],
                                                                          self._baseline_cv_scores.get(obj_name, np.nan))
            percent_better_than_baseline[obj_name] = percent_better

        pipeline_name = trained_pipeline.name
        pipeline_summary = trained_pipeline.summary
        pipeline_id = len(self._results['pipeline_results'])

        high_variance_cv_check = HighVarianceCVDataCheck(threshold=0.2)
        high_variance_cv_check_results = high_variance_cv_check.validate(pipeline_name=pipeline_name, cv_scores=cv_scores)
        high_variance_cv = False

        if high_variance_cv_check_results:
            logger.warning(high_variance_cv_check_results[0])
            high_variance_cv = True

        self._results['pipeline_results'][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline_name,
            "pipeline_class": type(trained_pipeline),
            "pipeline_summary": pipeline_summary,
            "parameters": parameters,
            "score": cv_score,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data,
            "percent_better_than_baseline_all_objectives": percent_better_than_baseline,
            "percent_better_than_baseline": percent_better_than_baseline[self.objective.name],
            "validation_score": cv_scores[0]
        }
        self._results['search_order'].append(pipeline_id)

        if self.add_result_callback:
            self.add_result_callback(self._results['pipeline_results'][pipeline_id], trained_pipeline, self)

    def _evaluate(self, pipeline, X, y):
        parameters = pipeline.parameters
        evaluation_results = self._compute_cv_scores(pipeline, X, y)
        logger.debug('Adding results for pipeline {}\nparameters {}\nevaluation_results {}'.format(pipeline.name, parameters, evaluation_results))

        self._add_result(trained_pipeline=pipeline,
                         parameters=parameters,
                         training_time=evaluation_results['training_time'],
                         cv_data=evaluation_results['cv_data'],
                         cv_scores=evaluation_results['cv_scores'])
        logger.debug('Adding results complete')
        return evaluation_results

    def get_pipeline(self, pipeline_id, random_state=0):
        """Given the ID of a pipeline training result, returns an untrained instance of the specified pipeline
        initialized with the parameters used to train that pipeline during automl search.

        Arguments:
            pipeline_id (int): pipeline to retrieve
            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

        Returns:
            PipelineBase: untrained pipeline instance associated with the provided ID
        """
        pipeline_results = self.results['pipeline_results'].get(pipeline_id)
        if pipeline_results is None:
            raise PipelineNotFoundError("Pipeline not found in automl results")
        pipeline_class = pipeline_results.get('pipeline_class')
        parameters = pipeline_results.get('parameters')
        if pipeline_class is None or parameters is None:
            raise PipelineNotFoundError("Pipeline class or parameters not found in automl results")
        return pipeline_class(parameters, random_state=random_state)

    def describe_pipeline(self, pipeline_id, return_dict=False):
        """Describe a pipeline

        Arguments:
            pipeline_id (int): pipeline to describe
            return_dict (bool): If True, return dictionary of information
                about pipeline. Defaults to False.

        Returns:
            Description of specified pipeline. Includes information such as
            type of pipeline components, problem, training time, cross validation, etc.
        """
        if pipeline_id not in self._results['pipeline_results']:
            raise PipelineNotFoundError("Pipeline not found")

        pipeline = self.get_pipeline(pipeline_id)
        pipeline_results = self._results['pipeline_results'][pipeline_id]

        pipeline.describe()
        log_subtitle(logger, "Training")
        logger.info("Training for {} problems.".format(pipeline.problem_type))

        if self.optimize_thresholds and self.objective.is_defined_for_problem_type(ProblemTypes.BINARY) and self.objective.can_optimize_threshold:
            logger.info("Objective to optimize binary classification pipeline thresholds for: {}".format(self.objective))

        logger.info("Total training time (including CV): %.1f seconds" % pipeline_results["training_time"])
        log_subtitle(logger, "Cross Validation", underline="-")

        all_objective_scores = [fold["all_objective_scores"] for fold in pipeline_results["cv_data"]]
        all_objective_scores = pd.DataFrame(all_objective_scores)

        for c in all_objective_scores:
            if c in ["# Training", "# Testing"]:
                all_objective_scores[c] = all_objective_scores[c].astype("object")
                continue

            mean = all_objective_scores[c].mean(axis=0)
            std = all_objective_scores[c].std(axis=0)
            all_objective_scores.loc["mean", c] = mean
            all_objective_scores.loc["std", c] = std
            all_objective_scores.loc["coef of var", c] = std / mean if abs(mean) > 0 else np.inf

        all_objective_scores = all_objective_scores.fillna("-")

        with pd.option_context('display.float_format', '{:.3f}'.format, 'expand_frame_repr', False):
            logger.info(all_objective_scores)

        if return_dict:
            return pipeline_results

    def add_to_rankings(self, pipeline, X, y):
        """Fits and evaluates a given pipeline then adds the results to the automl rankings with the requirement that automl search has been run.
        Please use the same data as previous runs of automl search. If pipeline already exists in rankings this method will return `None`.

        Arguments:
            pipeline (PipelineBase): pipeline to train and evaluate.

            X (pd.DataFrame): the input training data of shape [n_samples, n_features].

            y (pd.Series): the target training data of length [n_samples].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        self._set_data_split(X)

        pipeline_rows = self.full_rankings[self.full_rankings['pipeline_name'] == pipeline.name]
        for parameter in pipeline_rows['parameters']:
            if pipeline.parameters == parameter:
                return
        self._evaluate(pipeline, X, y)

    @property
    def results(self):
        """Class that allows access to a copy of the results from `automl_search`.

           Returns: dict containing `pipeline_results`: a dict with results from each pipeline,
                    and `search_order`: a list describing the order the pipelines were searched.
           """
        return copy.deepcopy(self._results)

    @property
    def has_searched(self):
        """Returns `True` if search has been ran and `False` if not"""
        searched = True if self._results['pipeline_results'] else False
        return searched

    @property
    def rankings(self):
        """Returns a pandas.DataFrame with scoring results from the highest-scoring set of parameters used with each pipeline."""
        return self.full_rankings.drop_duplicates(subset="pipeline_name", keep="first")

    @property
    def full_rankings(self):
        """Returns a pandas.DataFrame with scoring results from all pipelines searched"""
        ascending = True
        if self.objective.greater_is_better:
            ascending = False

        full_rankings_cols = ["id", "pipeline_name", "score", "validation_score",
                              "percent_better_than_baseline", "high_variance_cv", "parameters"]
        if not self.has_searched:
            return pd.DataFrame(columns=full_rankings_cols)

        rankings_df = pd.DataFrame(self._results['pipeline_results'].values())
        rankings_df = rankings_df[full_rankings_cols]
        rankings_df.sort_values("score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)
        return rankings_df

    @property
    def best_pipeline(self):
        """Returns an untrained instance of the best pipeline and parameters found during automl search.

        Returns:
            PipelineBase: untrained pipeline instance associated with the best automl search result.
        """
        if not self.has_searched:
            raise PipelineNotFoundError("automl search must be run before selecting `best_pipeline`.")

        best = self.rankings.iloc[0]
        return self.get_pipeline(best["id"])

    def save(self, file_path, pickle_protocol=cloudpickle.DEFAULT_PROTOCOL):
        """Saves AutoML object at file path

        Arguments:
            file_path (str): location to save file
            pickle_protocol (int): the pickle data stream format.

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f, protocol=pickle_protocol)

    @staticmethod
    def load(file_path):
        """Loads AutoML object at file path

        Arguments:
            file_path (str): location to find file to load

        Returns:
            AutoSearchBase object
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)
