import copy
import time
from collections import defaultdict

import cloudpickle
import numpy as np
import pandas as pd
import woodwork as ww
from sklearn.model_selection import BaseCrossValidator

from .pipeline_search_plots import PipelineSearchPlots

from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.callbacks import log_error_callback
from evalml.automl.engine import SequentialEngine
from evalml.automl.utils import (
    check_all_pipeline_names_unique,
    get_default_primary_search_objective,
    make_data_splitter
)
from evalml.exceptions import AutoMLSearchException, PipelineNotFoundError
from evalml.model_family import ModelFamily
from evalml.objectives import (
    get_core_objectives,
    get_non_core_objectives,
    get_objective
)
from evalml.pipelines import (
    MeanBaselineRegressionPipeline,
    ModeBaselineBinaryPipeline,
    ModeBaselineMulticlassPipeline,
    TimeSeriesBaselineBinaryPipeline,
    TimeSeriesBaselineMulticlassPipeline,
    TimeSeriesBaselineRegressionPipeline
)
from evalml.pipelines.components.utils import get_estimators
from evalml.pipelines.utils import make_pipeline
from evalml.preprocessing import split_data
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.tuners import SKOptTuner
from evalml.utils import convert_to_seconds, infer_feature_types
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

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    def __init__(self,
                 X_train=None,
                 y_train=None,
                 problem_type=None,
                 objective='auto',
                 max_iterations=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 data_splitter=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 error_callback=None,
                 additional_objectives=None,
                 random_seed=0,
                 n_jobs=-1,
                 tuner_class=None,
                 optimize_thresholds=True,
                 ensembling=False,
                 max_batches=None,
                 problem_configuration=None,
                 train_best_pipeline=True,
                 pipeline_parameters=None,
                 _ensembling_split_size=0.2,
                 _pipelines_per_batch=5):
        """Automated pipeline search

        Arguments:
            X_train (pd.DataFrame, ww.DataTable): The input training data of shape [n_samples, n_features]. Required.

            y_train (pd.Series, ww.DataColumn): The target training data of length [n_samples]. Required for supervised learning tasks.

            problem_type (str or ProblemTypes): type of supervised learning problem. See evalml.problem_types.ProblemType.all_problem_types for a full list.

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

            data_splitter (sklearn.model_selection.BaseCrossValidator): Data splitting method to use. Defaults to StratifiedKFold.

            tuner_class: The tuner class to use. Defaults to SKOptTuner.

            optimize_thresholds (bool): Whether or not to optimize the binary pipeline threshold. Defaults to True.

            start_iteration_callback (callable): Function called before each pipeline training iteration.
                Callback function takes three positional parameters: The pipeline class, the pipeline parameters, and the AutoMLSearch object.

            add_result_callback (callable): Function called after each pipeline training iteration.
                Callback function takes three positional parameters: A dictionary containing the training results for the new pipeline, an untrained_pipeline containing the parameters used during training, and the AutoMLSearch object.

            error_callback (callable): Function called when `search()` errors and raises an Exception.
                Callback function takes three positional parameters: the Exception raised, the traceback, and the AutoMLSearch object.
                Must also accepts kwargs, so AutoMLSearch is able to pass along other appropriate parameters by default.
                Defaults to None, which will call `log_error_callback`.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_seed (int): Seed for the random number generator. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            ensembling (boolean): If True, runs ensembling in a separate batch after every allowed pipeline class has been iterated over.
                If the number of unique pipelines to search over per batch is one, ensembling will not run. Defaults to False.

            max_batches (int): The maximum number of batches of pipelines to search. Parameters max_time, and
                max_iterations have precedence over stopping the search.

            problem_configuration (dict, None): Additional parameters needed to configure the search. For example,
                in time series problems, values should be passed in for the gap and max_delay variables.

            train_best_pipeline (boolean): Whether or not to train the best pipeline before returning it. Defaults to True.

            pipeline_parameters (dict): A dict of the parameters used to initalize a pipeline with.

            _ensembling_split_size (float): The amount of the training data we'll set aside for training ensemble metalearners. Only used when ensembling is True.
                Must be between 0 and 1, exclusive. Defaults to 0.2

            _pipelines_per_batch (int): The number of pipelines to train for every batch after the first one.
                The first batch will train a baseline pipline + one of each pipeline family allowed in the search.
        """
        if X_train is None:
            raise ValueError('Must specify training data as a 2d array using the X_train argument')
        if y_train is None:
            raise ValueError('Must specify training data target values as a 1d vector using the y_train argument')
        try:
            self.problem_type = handle_problem_types(problem_type)
        except ValueError:
            raise ValueError('choose one of (binary, multiclass, regression) as problem_type')

        self.tuner_class = tuner_class or SKOptTuner
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.error_callback = error_callback or log_error_callback
        self.data_splitter = data_splitter
        self.optimize_thresholds = optimize_thresholds
        self.ensembling = ensembling
        if objective == 'auto':
            objective = get_default_primary_search_objective(self.problem_type.value)
        objective = get_objective(objective, return_instance=False)
        self.objective = self._validate_objective(objective)
        if self.data_splitter is not None and not issubclass(self.data_splitter.__class__, BaseCrossValidator):
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
        self.objective_name_to_class = {o.name: o for o in [self.objective] + self.additional_objectives}

        if not isinstance(max_time, (int, float, str, type(None))):
            raise TypeError(f"Parameter max_time must be a float, int, string or None. Received {type(max_time)} with value {str(max_time)}..")
        if isinstance(max_time, (int, float)) and max_time < 0:
            raise ValueError(f"Parameter max_time must be None or non-negative. Received {max_time}.")
        if max_batches is not None and max_batches < 0:
            raise ValueError(f"Parameter max_batches must be None or non-negative. Received {max_batches}.")
        if max_iterations is not None and max_iterations < 0:
            raise ValueError(f"Parameter max_iterations must be None or non-negative. Received {max_iterations}.")
        self.max_time = convert_to_seconds(max_time) if isinstance(max_time, str) else max_time
        self.max_iterations = max_iterations
        self.max_batches = max_batches
        self._pipelines_per_batch = _pipelines_per_batch
        if not self.max_iterations and not self.max_time and not self.max_batches:
            self.max_batches = 1
            logger.info("Using default limit of max_batches=1.\n")

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
        self.random_seed = random_seed
        self.n_jobs = n_jobs

        self.plot = None
        try:
            self.plot = PipelineSearchPlots(self)
        except ImportError:
            logger.warning("Unable to import plotly; skipping pipeline search plotting\n")

        self.allowed_pipelines = allowed_pipelines
        self.allowed_model_families = allowed_model_families
        self._automl_algorithm = None
        self._start = 0.0
        self._baseline_cv_scores = {}
        self.show_batch_output = False

        self._validate_problem_type()
        self.problem_configuration = self._validate_problem_configuration(problem_configuration)
        self._train_best_pipeline = train_best_pipeline
        self._best_pipeline = None
        self._searched = False

        self.X_train = infer_feature_types(X_train)
        self.y_train = infer_feature_types(y_train)
        self.ensembling_indices = None

        default_data_splitter = make_data_splitter(self.X_train, self.y_train, self.problem_type, self.problem_configuration,
                                                   n_splits=3, shuffle=True, random_seed=self.random_seed)
        self.data_splitter = self.data_splitter or default_data_splitter
        self.pipeline_parameters = pipeline_parameters if pipeline_parameters is not None else {}
        self.search_iteration_plot = None
        self._interrupted = False

        if self.allowed_pipelines is None:
            logger.info("Generating pipelines to search over...")
            allowed_estimators = get_estimators(self.problem_type, self.allowed_model_families)
            logger.debug(f"allowed_estimators set to {[estimator.name for estimator in allowed_estimators]}")
            self.allowed_pipelines = [make_pipeline(self.X_train, self.y_train, estimator, self.problem_type, custom_hyperparameters=self.pipeline_parameters) for estimator in allowed_estimators]

        if self.allowed_pipelines == []:
            raise ValueError("No allowed pipelines to search")
        check_all_pipeline_names_unique(self.allowed_pipelines)

        run_ensembling = self.ensembling
        if run_ensembling and len(self.allowed_pipelines) == 1:
            logger.warning("Ensembling is set to True, but the number of unique pipelines is one, so ensembling will not run.")
            run_ensembling = False

        if run_ensembling and self.max_iterations is not None:
            # Baseline + first batch + each pipeline iteration + 1
            first_ensembling_iteration = (1 + len(self.allowed_pipelines) + len(self.allowed_pipelines) * self._pipelines_per_batch + 1)
            if self.max_iterations < first_ensembling_iteration:
                run_ensembling = False
                logger.warning(f"Ensembling is set to True, but max_iterations is too small, so ensembling will not run. Set max_iterations >= {first_ensembling_iteration} to run ensembling.")
            else:
                logger.info(f"Ensembling will run at the {first_ensembling_iteration} iteration and every {len(self.allowed_pipelines) * self._pipelines_per_batch} iterations after that.")

        if self.max_batches and self.max_iterations is None:
            self.show_batch_output = True
            if run_ensembling:
                ensemble_nth_batch = len(self.allowed_pipelines) + 1
                num_ensemble_batches = (self.max_batches - 1) // ensemble_nth_batch
                if num_ensemble_batches == 0:
                    run_ensembling = False
                    logger.warning(f"Ensembling is set to True, but max_batches is too small, so ensembling will not run. Set max_batches >= {ensemble_nth_batch + 1} to run ensembling.")
                else:
                    logger.info(f"Ensembling will run every {ensemble_nth_batch} batches.")

                self.max_iterations = (1 + len(self.allowed_pipelines) +
                                       self._pipelines_per_batch * (self.max_batches - 1 - num_ensemble_batches) +
                                       num_ensemble_batches)
            else:
                self.max_iterations = 1 + len(self.allowed_pipelines) + (self._pipelines_per_batch * (self.max_batches - 1))
        if run_ensembling:
            if not (0 < _ensembling_split_size < 1):
                raise ValueError(f"Ensembling split size must be between 0 and 1 exclusive, received {_ensembling_split_size}")
            X_shape = ww.DataTable(np.arange(self.X_train.shape[0]))
            _, ensembling_indices, _, _ = split_data(X_shape, self.y_train, problem_type=self.problem_type, test_size=_ensembling_split_size, random_seed=self.random_seed)
            self.ensembling_indices = ensembling_indices.to_dataframe()[0].tolist()

        self._engine = SequentialEngine(self.X_train,
                                        self.y_train,
                                        self.ensembling_indices,
                                        self,
                                        should_continue_callback=self._should_continue,
                                        pre_evaluation_callback=self._pre_evaluation_callback,
                                        post_evaluation_callback=self._post_evaluation_callback)

        self.allowed_model_families = list(set([p.model_family for p in (self.allowed_pipelines)]))

        logger.debug(f"allowed_pipelines set to {[pipeline.name for pipeline in self.allowed_pipelines]}")
        logger.debug(f"allowed_model_families set to {self.allowed_model_families}")
        if len(self.problem_configuration):
            pipeline_params = {**{'pipeline': self.problem_configuration}, **self.pipeline_parameters}
        else:
            pipeline_params = self.pipeline_parameters

        self._automl_algorithm = IterativeAlgorithm(
            max_iterations=self.max_iterations,
            allowed_pipelines=self.allowed_pipelines,
            tuner_class=self.tuner_class,
            random_seed=self.random_seed,
            n_jobs=self.n_jobs,
            number_features=self.X_train.shape[1],
            pipelines_per_batch=self._pipelines_per_batch,
            ensembling=run_ensembling,
            pipeline_params=pipeline_params
        )

    def _pre_evaluation_callback(self, pipeline):
        if self.start_iteration_callback:
            self.start_iteration_callback(pipeline.__class__, pipeline.parameters, self)
        desc = f"{pipeline.name}"
        if len(desc) > AutoMLSearch._MAX_NAME_LEN:
            desc = desc[:AutoMLSearch._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(AutoMLSearch._MAX_NAME_LEN)
        batch_number = 1
        if self._automl_algorithm is not None and self._automl_algorithm.batch_number > 0:
            batch_number = self._automl_algorithm.batch_number
        update_pipeline(logger,
                        desc,
                        len(self._results['pipeline_results']) + 1,
                        self.max_iterations,
                        self._start,
                        batch_number,
                        self.show_batch_output)

    def _validate_objective(self, objective):
        non_core_objectives = get_non_core_objectives()
        if isinstance(objective, type):
            if objective in non_core_objectives:
                raise ValueError(f"{objective.name.lower()} is not allowed in AutoML! "
                                 "Use evalml.objectives.utils.get_core_objective_names() "
                                 "to get all objective names allowed in automl.")
            return objective()
        return objective

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
            rankings_str = self.rankings.drop(['parameters'], axis='columns').to_string()
            rankings_desc = f"\nSearch Results: \n{'='*20}\n{rankings_str}"

        return search_desc + rankings_desc

    def _validate_problem_configuration(self, problem_configuration=None):
        if self.problem_type in [ProblemTypes.TIME_SERIES_REGRESSION]:
            required_parameters = {'gap', 'max_delay'}
            if not problem_configuration or not all(p in problem_configuration for p in required_parameters):
                raise ValueError("user_parameters must be a dict containing values for at least the gap and max_delay "
                                 f"parameters. Received {problem_configuration}.")
        return problem_configuration or {}

    def _handle_keyboard_interrupt(self):
        """Presents a prompt to the user asking if they want to stop the search.

        Returns:
            bool: If True, search should terminate early
        """
        leading_char = "\n"
        start_of_loop = time.time()
        while True:
            choice = input(leading_char + "Do you really want to exit search (y/n)? ").strip().lower()
            if choice == "y":
                logger.info("Exiting AutoMLSearch.")
                return True
            elif choice == "n":
                # So that the time in this loop does not count towards the time budget (if set)
                time_in_loop = time.time() - start_of_loop
                self._start += time_in_loop
                return False
            else:
                leading_char = ""

    def search(self, show_iteration_plot=True):
        """Find the best pipeline for the data set.

        Arguments:
            feature_types (list, optional): list of feature types, either numerical or categorical.
                Categorical features will automatically be encoded

            show_iteration_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.
        """
        if self._searched:
            logger.info("AutoMLSearch.search() has already been run and will not run again on the same instance. Re-initialize AutoMLSearch to search again.")
            return

        # don't show iteration plot outside of a jupyter notebook
        if show_iteration_plot:
            try:
                get_ipython
            except NameError:
                show_iteration_plot = False

        log_title(logger, "Beginning pipeline search")
        logger.info("Optimizing for %s. " % self.objective.name)
        logger.info("{} score is better.\n".format('Greater' if self.objective.greater_is_better else 'Lower'))
        logger.info(f"Using {self._engine.__class__.__name__} to train and score pipelines.")

        if self.max_batches is not None:
            logger.info(f"Searching up to {self.max_batches} batches for a total of {self.max_iterations} pipelines. ")
        elif self.max_iterations is not None:
            logger.info("Searching up to %s pipelines. " % self.max_iterations)
        if self.max_time is not None:
            logger.info("Will stop searching for new pipelines after %d seconds.\n" % self.max_time)
        logger.info("Allowed model families: %s\n" % ", ".join([model.value for model in self.allowed_model_families]))
        self.search_iteration_plot = None
        if self.plot:
            self.search_iteration_plot = self.plot.search_iteration_plot(interactive_plot=show_iteration_plot)

        self._start = time.time()

        try:
            self._add_baseline_pipelines()
        except KeyboardInterrupt:
            if self._handle_keyboard_interrupt():
                self._interrupted = True

        current_batch_pipelines = []
        current_batch_pipeline_scores = []
        new_pipeline_ids = []
        loop_interrupted = False
        while self._should_continue():
            try:
                if not loop_interrupted:
                    current_batch_pipelines = self._automl_algorithm.next_batch()
            except StopIteration:
                logger.info('AutoML Algorithm out of recommendations, ending')
                break
            try:
                new_pipeline_ids = self._engine.evaluate_batch(current_batch_pipelines)
                loop_interrupted = False
            except KeyboardInterrupt:
                loop_interrupted = True
                if self._handle_keyboard_interrupt():
                    break
            full_rankings = self.full_rankings
            current_batch_idx = full_rankings['id'].isin(new_pipeline_ids)
            current_batch_pipeline_scores = full_rankings[current_batch_idx]['score']
            if len(current_batch_pipeline_scores) and current_batch_pipeline_scores.isna().all():
                raise AutoMLSearchException(f"All pipelines in the current AutoML batch produced a score of np.nan on the primary objective {self.objective}.")

        self.search_duration = time.time() - self._start
        elapsed_time = time_elapsed(self._start)
        desc = f"\nSearch finished after {elapsed_time}"
        desc = desc.ljust(self._MAX_NAME_LEN)
        logger.info(desc)

        self._find_best_pipeline()
        if self._best_pipeline is not None:
            best_pipeline = self.rankings.iloc[0]
            best_pipeline_name = best_pipeline["pipeline_name"]
            logger.info(f"Best pipeline: {best_pipeline_name}")
            logger.info(f"Best pipeline {self.objective.name}: {best_pipeline['score']:3f}")
        self._searched = True

    def _find_best_pipeline(self):
        """Finds the best pipeline in the rankings
        If self._best_pipeline already exists, check to make sure it is different from the current best pipeline before training and thresholding"""
        if len(self.rankings) == 0:
            return
        best_pipeline = self.rankings.iloc[0]
        if not (self._best_pipeline and self._best_pipeline == self.get_pipeline(best_pipeline['id'])):
            best_pipeline = self.get_pipeline(best_pipeline['id'])
            if self._train_best_pipeline:
                if best_pipeline.model_family == ModelFamily.ENSEMBLE:
                    X_train, y_train = self.X_train.iloc[self.ensembling_indices], self.y_train.iloc[self.ensembling_indices]
                else:
                    X_train = self.X_train
                    y_train = self.y_train
                if hasattr(self.data_splitter, "transform_sample"):
                    train_indices = self.data_splitter.transform_sample(X_train, y_train)
                    X_train = X_train.iloc[train_indices]
                    y_train = y_train.iloc[train_indices]
                best_pipeline = self._engine.train_pipeline(best_pipeline, X_train, y_train,
                                                            self.optimize_thresholds, self.objective)
            self._best_pipeline = best_pipeline

    def _num_pipelines(self):
        """Return the number of pipeline evaluations which have been made

        Returns:
            int: the number of pipeline evaluations made in the search
        """
        return len(self._results['pipeline_results'])

    def _should_continue(self):
        """Given the original stopping criterion and current state, should the search continue?

        Returns:
            bool: True if yes, False if no.
        """
        if self._interrupted:
            return False

        # for add_to_rankings
        if self._searched:
            return True

        # Run at least one pipeline for every search
        num_pipelines = self._num_pipelines()
        if num_pipelines == 0:
            return True

        # check max_time and max_iterations
        elapsed = time.time() - self._start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_iterations and num_pipelines >= self.max_iterations:
            return False

        # check for early stopping
        if self.patience is None or self.tolerance is None:
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
        return True

    def _validate_problem_type(self):
        for obj in self.additional_objectives:
            if not obj.is_defined_for_problem_type(self.problem_type):
                raise ValueError("Additional objective {} is not compatible with a {} problem.".format(obj.name, self.problem_type.value))

        for pipeline in self.allowed_pipelines or []:
            if pipeline.problem_type != self.problem_type:
                raise ValueError("Given pipeline {} is not compatible with problem_type {}.".format(pipeline.name, self.problem_type.value))

    def _add_baseline_pipelines(self):
        """Fits a baseline pipeline to the data.

        This is the first pipeline fit during search.
        """
        if self.problem_type == ProblemTypes.BINARY:
            baseline = ModeBaselineBinaryPipeline(parameters={})
        elif self.problem_type == ProblemTypes.MULTICLASS:
            baseline = ModeBaselineMulticlassPipeline(parameters={})
        elif self.problem_type == ProblemTypes.REGRESSION:
            baseline = MeanBaselineRegressionPipeline(parameters={})
        else:
            pipeline_class = {ProblemTypes.TIME_SERIES_REGRESSION: TimeSeriesBaselineRegressionPipeline,
                              ProblemTypes.TIME_SERIES_MULTICLASS: TimeSeriesBaselineMulticlassPipeline,
                              ProblemTypes.TIME_SERIES_BINARY: TimeSeriesBaselineBinaryPipeline}[self.problem_type]
            gap = self.problem_configuration['gap']
            max_delay = self.problem_configuration['max_delay']
            baseline = pipeline_class(parameters={"pipeline": {"gap": gap, "max_delay": max_delay},
                                                  "Time Series Baseline Estimator": {"gap": gap, "max_delay": max_delay}})
        self._engine.evaluate_batch([baseline])

    @staticmethod
    def _get_mean_cv_scores_for_all_objectives(cv_data, objective_name_to_class):
        scores = defaultdict(int)
        n_folds = len(cv_data)
        for fold_data in cv_data:
            for field, value in fold_data['all_objective_scores'].items():
                # The 'all_objective_scores' field contains scores for all objectives
                # but also fields like "# Training" and "# Testing", so we want to exclude them since
                # they are not scores
                if field in objective_name_to_class:
                    scores[field] += value
        return {objective: float(score) / n_folds for objective, score in scores.items()}

    def _post_evaluation_callback(self, pipeline, evaluation_results):
        training_time = evaluation_results['training_time']
        cv_data = evaluation_results['cv_data']
        cv_scores = evaluation_results['cv_scores']
        is_baseline = pipeline.model_family == ModelFamily.BASELINE
        cv_score = cv_scores.mean()

        percent_better_than_baseline = {}
        mean_cv_all_objectives = self._get_mean_cv_scores_for_all_objectives(cv_data, self.objective_name_to_class)
        if is_baseline:
            self._baseline_cv_scores = mean_cv_all_objectives
        for obj_name in mean_cv_all_objectives:
            objective_class = self.objective_name_to_class[obj_name]

            # In the event add_to_rankings is called before search _baseline_cv_scores will be empty so we will return
            # nan for the base score.
            percent_better = objective_class.calculate_percent_difference(mean_cv_all_objectives[obj_name],
                                                                          self._baseline_cv_scores.get(obj_name, np.nan))
            percent_better_than_baseline[obj_name] = percent_better

        high_variance_cv = self._check_for_high_variance(pipeline, cv_scores)

        pipeline_id = len(self._results['pipeline_results'])
        self._results['pipeline_results'][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline.name,
            "pipeline_class": type(pipeline),
            "pipeline_summary": pipeline.summary,
            "parameters": pipeline.parameters,
            "score": cv_score,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data,
            "percent_better_than_baseline_all_objectives": percent_better_than_baseline,
            "percent_better_than_baseline": percent_better_than_baseline[self.objective.name],
            "validation_score": cv_scores[0]
        }

        if pipeline.model_family == ModelFamily.ENSEMBLE:
            input_pipeline_ids = [self._automl_algorithm._best_pipeline_info[model_family]["id"] for model_family in self._automl_algorithm._best_pipeline_info]
            self._results['pipeline_results'][pipeline_id]["input_pipeline_ids"] = input_pipeline_ids

        self._results['search_order'].append(pipeline_id)

        if not is_baseline:
            score_to_minimize = -cv_score if self.objective.greater_is_better else cv_score
            try:
                self._automl_algorithm.add_result(score_to_minimize, pipeline, self._results['pipeline_results'][pipeline_id])
            except PipelineNotFoundError:
                pass

        if self.search_iteration_plot:
            self.search_iteration_plot.update()

        if self.add_result_callback:
            self.add_result_callback(self._results['pipeline_results'][pipeline_id], pipeline, self)
        return pipeline_id

    def _check_for_high_variance(self, pipeline, cv_scores, threshold=0.2):
        """Checks cross-validation scores and logs a warning if variance is higher than specified threshhold."""
        pipeline_name = pipeline.name

        high_variance_cv = False
        cv_scores_std = cv_scores.std()
        cv_scores_mean = cv_scores.mean()
        if cv_scores_std != 0 and cv_scores_mean != 0:
            high_variance_cv = bool(abs(cv_scores_std / cv_scores_mean) > threshold)
        if high_variance_cv:
            logger.warning(f"High coefficient of variation (cv >= {threshold}) within cross validation scores. {pipeline_name} may not perform as estimated on unseen data.")
        return high_variance_cv

    def get_pipeline(self, pipeline_id):
        """Given the ID of a pipeline training result, returns an untrained instance of the specified pipeline
        initialized with the parameters used to train that pipeline during automl search.

        Arguments:
            pipeline_id (int): pipeline to retrieve

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
        return pipeline_class(parameters, random_seed=self.random_seed)

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

        if pipeline.model_family == ModelFamily.ENSEMBLE:
            logger.info("Input for ensembler are pipelines with IDs: " + str(pipeline_results['input_pipeline_ids']))

        log_subtitle(logger, "Training")
        logger.info("Training for {} problems.".format(pipeline.problem_type))

        if self.optimize_thresholds and self.objective.is_defined_for_problem_type(ProblemTypes.BINARY) and self.objective.can_optimize_threshold:
            logger.info("Objective to optimize binary classification pipeline thresholds for: {}".format(self.objective))

        logger.info("Total training time (including CV): %.1f seconds" % pipeline_results["training_time"])
        log_subtitle(logger, "Cross Validation", underline="-")

        all_objective_scores = [fold["all_objective_scores"] for fold in pipeline_results["cv_data"]]
        all_objective_scores = pd.DataFrame(all_objective_scores)

        for c in all_objective_scores:
            if c in ["# Training", "# Validation"]:
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

    def add_to_rankings(self, pipeline):
        """Fits and evaluates a given pipeline then adds the results to the automl rankings with the requirement that automl search has been run.

        Arguments:
            pipeline (PipelineBase): pipeline to train and evaluate.
        """
        pipeline_rows = self.full_rankings[self.full_rankings['pipeline_name'] == pipeline.name]
        for parameter in pipeline_rows['parameters']:
            if pipeline.parameters == parameter:
                return

        self._engine.evaluate_batch([pipeline])
        self._find_best_pipeline()

    @property
    def results(self):
        """Class that allows access to a copy of the results from `automl_search`.

           Returns: dict containing `pipeline_results`: a dict with results from each pipeline,
                    and `search_order`: a list describing the order the pipelines were searched.
           """
        return copy.deepcopy(self._results)

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
        if not self._results['pipeline_results']:
            return pd.DataFrame(columns=full_rankings_cols)

        rankings_df = pd.DataFrame(self._results['pipeline_results'].values())
        rankings_df = rankings_df[full_rankings_cols]
        rankings_df.sort_values("score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)
        return rankings_df

    @property
    def best_pipeline(self):
        """Returns a trained instance of the best pipeline and parameters found during automl search. If `train_best_pipeline` is set to False, returns an untrained pipeline instance.

        Returns:
            PipelineBase: A trained instance of the best pipeline and parameters found during automl search. If `train_best_pipeline` is set to False, returns an untrained pipeline instance.
        """
        if not self._best_pipeline:
            raise PipelineNotFoundError("automl search must be run before selecting `best_pipeline`.")

        return self._best_pipeline

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

    def train_pipelines(self, pipelines):
        """Train a list of pipelines on the training data.

        This can be helpful for training pipelines once the search is complete.

        Arguments:
            pipelines (list(PipelineBase)): List of pipelines to train.

        Returns:
            Dict[str, PipelineBase]: Dictionary keyed by pipeline name that maps to the fitted pipeline.
            Note that the any pipelines that error out during training will not be included in the dictionary
            but the exception and stacktrace will be displayed in the log.
        """
        return self._engine.train_batch(pipelines)

    def score_pipelines(self, pipelines, X_holdout, y_holdout, objectives):
        """Score a list of pipelines on the given holdout data.

        Arguments:
            pipelines (list(PipelineBase)): List of pipelines to train.
            X_holdout (ww.DataTable, pd.DataFrame): Holdout features.
            y_holdout (ww.DataTable, pd.DataFrame): Holdout targets for scoring.
            objectives (list(str), list(ObjectiveBase)): Objectives used for scoring.

        Returns:
            Dict[str, Dict[str, float]]: Dictionary keyed by pipeline name that maps to a dictionary of scores.
            Note that the any pipelines that error out during scoring will not be included in the dictionary
            but the exception and stacktrace will be displayed in the log.
        """
        return self._engine.score_batch(pipelines, X_holdout, y_holdout, objectives)
