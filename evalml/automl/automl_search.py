import copy
import time
import warnings
from collections import OrderedDict

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    BaseCrossValidator,
    KFold,
    StratifiedKFold,
    train_test_split
)

from .pipeline_search_plots import PipelineSearchPlots

from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.data_splitters import TrainingValidationSplit
from evalml.data_checks import DataChecks, DefaultDataChecks, EmptyDataChecks
from evalml.data_checks.data_check_message_type import DataCheckMessageType
from evalml.exceptions import (
    AutoMLSearchException,
    PipelineNotFoundError,
    PipelineScoreError
)
from evalml.objectives import (
    CostBenefitMatrix,
    FraudCost,
    LeadScoring,
    MeanSquaredLogError,
    Recall,
    RecallMacro,
    RecallMicro,
    RecallWeighted,
    RootMeanSquaredLogError,
    get_objective,
    get_objectives
)
from evalml.objectives.utils import (
    _all_objectives_dict,
    _print_objectives_in_table
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
from evalml.utils.gen_utils import classproperty
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

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    _DEFAULT_OBJECTIVES = {'binary': 'Log Loss Binary',
                           'multiclass': 'Log Loss Multiclass',
                           'regression': 'R2'}

    def __init__(self,
                 problem_type=None,
                 objective='auto',
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 data_split=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 tuner_class=None,
                 verbose=True,
                 optimize_thresholds=False,
                 _max_batches=None):
        """Automated pipeline search

        Arguments:
            problem_type (str or ProblemTypes): Choice of 'regression', 'binary', or 'multiclass', depending on the desired problem type.

            objective (str, ObjectiveBase): The objective to optimize for. When set to auto, chooses:
                LogLossBinary for binary classification problems,
                LogLossMulticlass for multiclass classification problems, and
                R2 for regression problems.

            max_pipelines (int): Maximum number of pipelines to search. If max_pipelines and
                max_time is not set, then max_pipelines will default to max_pipelines of 5.

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

            data_split (sklearn.model_selection.BaseCrossValidator): data splitting method to use. Defaults to StratifiedKFold.

            tuner_class: the tuner class to use. Defaults to scikit-optimize tuner

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed three parameters: pipeline_class, parameters, and the AutoMLSearch object.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed three parameters: A dictionary containing the training results for the new pipeline, an untrained_pipeline containing the parameters used during training, and the AutoMLSearch object.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            verbose (boolean): If True, turn verbosity on. Defaults to True

            _max_batches (int): The maximum number of batches of pipelines to search. Parameters max_time, and
                max_pipelines have precedence over stopping the search.
        """
        try:
            self.problem_type = handle_problem_types(problem_type)
        except ValueError:
            raise ValueError('choose one of (binary, multiclass, regression) as problem_type')

        self.tuner_class = tuner_class or SKOptTuner
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.data_split = data_split
        self.verbose = verbose
        self.optimize_thresholds = optimize_thresholds
        if objective == 'auto':
            objective = self._DEFAULT_OBJECTIVES[self.problem_type.value]
        objective = get_objective(objective, return_instance=False)
        self.objective = self._validate_objective(objective)
        if self.data_split is not None and not issubclass(self.data_split.__class__, BaseCrossValidator):
            raise ValueError("Not a valid data splitter")
        if self.problem_type != self.objective.problem_type:
            raise ValueError("Given objective {} is not compatible with a {} problem.".format(self.objective.name, self.problem_type.value))
        if additional_objectives is None:
            additional_objectives = [obj for obj in get_objectives(self.problem_type) if obj not in self._objectives_not_allowed_in_automl]
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

        self.max_pipelines = max_pipelines
        if self.max_pipelines is None and self.max_time is None and _max_batches is None:
            self.max_pipelines = 5
            logger.info("Using default limit of max_pipelines=5.\n")

        if patience and (not isinstance(patience, int) or patience < 0):
            raise ValueError("patience value must be a positive integer. Received {} instead".format(patience))

        if tolerance and (tolerance > 1.0 or tolerance < 0.0):
            raise ValueError("tolerance value must be a float between 0.0 and 1.0 inclusive. Received {} instead".format(tolerance))

        self.patience = patience
        self.tolerance = tolerance or 0.0
        self._results = {
            'pipeline_results': {},
            'search_order': []
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
        self._baseline_cv_score = None

        if _max_batches is not None and _max_batches <= 0:
            raise ValueError("Parameter max batches must be None or non-negative. Received {max_batches}.")
        self._max_batches = _max_batches
        # This is the default value for IterativeAlgorithm - setting this explicitly makes sure that
        # the behavior of max_batches does not break if IterativeAlgorithm is changed.
        self._pipelines_per_batch = 5

        self._validate_problem_type()

    @classproperty
    def _objectives_not_allowed_in_automl(self):
        return {CostBenefitMatrix, FraudCost, LeadScoring,
                MeanSquaredLogError, Recall, RecallMacro, RecallMicro, RecallWeighted, RootMeanSquaredLogError}

    @classmethod
    def print_objective_names_allowed_in_automl(cls):
        names = [name for name, value in _all_objectives_dict().items() if value not in cls._objectives_not_allowed_in_automl]
        _print_objectives_in_table(names)

    def _validate_objective(self, objective):
        if isinstance(objective, type):
            if objective in self._objectives_not_allowed_in_automl:
                raise ValueError(f"{objective.name} is not allowed in AutoML! "
                                 "Use evalml.automl.AutoMLSearch.print_objective_names_allowed_in_automl() "
                                 "to get all objective names allowed in automl.")
            return objective()
        return objective

    @property
    def data_check_results(self):
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
            f"Max Pipelines: {self.max_pipelines}\n"
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

    @staticmethod
    def _validate_data_checks(data_checks):
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
            return DataChecks(data_checks)
        elif isinstance(data_checks, str):
            if data_checks == "auto":
                return DefaultDataChecks()
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

        Args:
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

    def search(self, X, y, data_checks="auto", feature_types=None, show_iteration_plot=True):
        """Find the best pipeline for the data set.

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types, either numerical or categorical.
                Categorical features will automatically be encoded

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

        # make everything pandas objects
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Set the default data splitter
        if self.problem_type == ProblemTypes.REGRESSION:
            default_data_split = KFold(n_splits=3, random_state=self.random_state)
        elif self.problem_type in [ProblemTypes.BINARY, ProblemTypes.MULTICLASS]:
            default_data_split = StratifiedKFold(n_splits=3, random_state=self.random_state)

        if X.shape[0] > self._LARGE_DATA_ROW_THRESHOLD:
            default_data_split = TrainingValidationSplit(test_size=0.25)

        self.data_split = self.data_split or default_data_split

        data_checks = self._validate_data_checks(data_checks)
        data_check_results = data_checks.validate(X, y)

        if len(data_check_results) > 0:
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
            self.allowed_pipelines = [make_pipeline(X, y, estimator, self.problem_type) for estimator in allowed_estimators]

        if self.allowed_pipelines == []:
            raise ValueError("No allowed pipelines to search")
        if self._max_batches and self.max_pipelines is None:
            self.max_pipelines = 1 + len(self.allowed_pipelines) + (self._pipelines_per_batch * (self._max_batches - 1))

        self.allowed_model_families = list(set([p.model_family for p in (self.allowed_pipelines)]))

        logger.debug(f"allowed_pipelines set to {[pipeline.name for pipeline in self.allowed_pipelines]}")
        logger.debug(f"allowed_model_families set to {self.allowed_model_families}")

        self._automl_algorithm = IterativeAlgorithm(
            max_pipelines=self.max_pipelines,
            allowed_pipelines=self.allowed_pipelines,
            tuner_class=self.tuner_class,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            number_features=X.shape[1],
            pipelines_per_batch=self._pipelines_per_batch
        )

        log_title(logger, "Beginning pipeline search")
        logger.info("Optimizing for %s. " % self.objective.name)
        logger.info("{} score is better.\n".format('Greater' if self.objective.greater_is_better else 'Lower'))

        if self.max_pipelines is not None:
            logger.info("Searching up to %s pipelines. " % self.max_pipelines)
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

                update_pipeline(logger, desc, len(self._results['pipeline_results']) + 1, self.max_pipelines, self._start)

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

        # check max_time and max_pipelines
        elapsed = time.time() - start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_pipelines and num_pipelines >= self.max_pipelines:
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
            if obj.problem_type != self.problem_type:
                raise ValueError("Additional objective {} is not compatible with a {} problem.".format(obj.name, self.problem_type.value))

        for pipeline in self.allowed_pipelines or []:
            if not pipeline.problem_type == self.problem_type:
                raise ValueError("Given pipeline {} is not compatible with problem_type {}.".format(pipeline.name, self.problem_type.value))

    def _add_baseline_pipelines(self, X, y):
        """Fits a baseline pipeline to the data.

        This is the first pipeline fit during search.

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]
            y (pd.Series): the target training labels of length [n_samples]

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

                update_pipeline(logger, desc, len(self._results['pipeline_results']) + 1, self.max_pipelines,
                                self._start)

                baseline_results = self._compute_cv_scores(baseline, X, y)
                self._baseline_cv_score = baseline_results["cv_score_mean"]
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

    def _compute_cv_scores(self, pipeline, X, y):
        start = time.time()
        cv_data = []
        logger.info("\tStarting cross validation")
        for i, (train, test) in enumerate(self.data_split.split(X, y)):
            logger.debug(f"\t\tTraining and scoring on fold {i}")
            X_train, X_test = X.iloc[train], X.iloc[test]
            y_train, y_test = y.iloc[train], y.iloc[test]
            objectives_to_score = [self.objective] + self.additional_objectives
            cv_pipeline = None
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None
                if self.optimize_thresholds and self.objective.problem_type == ProblemTypes.BINARY and self.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state)
                cv_pipeline = pipeline.clone()
                logger.debug(f"\t\t\tFold {i}: starting training")
                cv_pipeline.fit(X_train, y_train)
                logger.debug(f"\t\t\tFold {i}: finished training")
                if self.objective.problem_type == ProblemTypes.BINARY:
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
                if isinstance(e, PipelineScoreError):
                    logger.info(f"\t\t\tFold {i}: Encountered an error scoring the following objectives: {', '.join(e.exceptions)}.")
                    logger.info(f"\t\t\tFold {i}: The scores for these objectives will be replaced with nan.")
                    logger.info(f"\t\t\tFold {i}: Please check {logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
                    logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
                    logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
                    nan_scores = {objective: np.nan for objective in e.exceptions}
                    scores = {**nan_scores, **e.scored_successfully}
                    scores = OrderedDict({o.name: scores[o.name] for o in [self.objective] + self.additional_objectives})
                    score = scores[self.objective.name]
                else:
                    logger.info(f"\t\t\tFold {i}: Encountered an error.")
                    logger.info(f"\t\t\tFold {i}: All scores will be replaced with nan.")
                    logger.info(f"\t\t\tFold {i}: Please check {logger.handlers[1].baseFilename} for the current hyperparameters and stack trace.")
                    logger.debug(f"\t\t\tFold {i}: Hyperparameters:\n\t{pipeline.hyperparameters}")
                    logger.debug(f"\t\t\tFold {i}: Exception during automl search: {str(e)}")
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
        percent_better = self.objective.calculate_percent_difference(cv_score, self._baseline_cv_score)
        # calculate high_variance_cv
        # if the coefficient of variance is greater than .2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high_variance_cv = (cv_scores.std() / cv_scores.mean()) > .2

        pipeline_name = trained_pipeline.name
        pipeline_summary = trained_pipeline.summary
        pipeline_id = len(self._results['pipeline_results'])

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
            "percent_better_than_baseline": percent_better
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

        if self.optimize_thresholds and self.objective.problem_type == ProblemTypes.BINARY and self.objective.can_optimize_threshold:
            logger.info("Objective to optimize binary classification pipeline thresholds for: {}".format(self.objective))

        logger.info("Total training time (including CV): %.1f seconds" % pipeline_results["training_time"])
        log_subtitle(logger, "Cross Validation", underline="-")

        if pipeline_results["high_variance_cv"]:
            logger.warning("High variance within cross validation scores. " +
                           "Model may not perform as estimated on unseen data.")

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

            y (pd.Series): the target training labels of length [n_samples].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if not self.has_searched:
            raise RuntimeError("Please run automl search before calling `add_to_rankings()`")

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

        full_rankings_cols = ["id", "pipeline_name", "score", "percent_better_than_baseline",
                              "high_variance_cv", "parameters"]
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
