import time
import warnings
from collections import OrderedDict
from sys import stdout

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tqdm import tqdm

from .pipeline_search_plots import PipelineSearchPlots

from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.data_checks import DataChecks, DefaultDataChecks
from evalml.data_checks.data_check_message_type import DataCheckMessageType
from evalml.objectives import get_objective, get_objectives
from evalml.pipelines import (
    MeanBaselineRegressionPipeline,
    ModeBaselineBinaryPipeline,
    ModeBaselineMulticlassPipeline
)
from evalml.pipelines.utils import get_estimators, make_pipeline
from evalml.problem_types import ProblemTypes, handle_problem_types
from evalml.tuners import SKOptTuner
from evalml.utils import convert_to_seconds, get_random_state
from evalml.utils.logger import get_logger, log_subtitle, log_title

logger = get_logger(__file__)


class AutoMLSearch:
    """Automated Pipeline search."""
    _MAX_NAME_LEN = 40

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    _DEFAULT_OBJECTIVES = {'binary': 'log_loss_binary',
                           'multiclass': 'log_loss_multi',
                           'regression': 'r2'}

    def __init__(self,
                 problem_type=None,
                 objective='auto',
                 max_pipelines=None,
                 max_time=None,
                 patience=None,
                 tolerance=None,
                 cv=None,
                 allowed_pipelines=None,
                 allowed_model_families=None,
                 start_iteration_callback=None,
                 add_result_callback=None,
                 additional_objectives=None,
                 random_state=0,
                 n_jobs=-1,
                 tuner_class=None,
                 verbose=True,
                 optimize_thresholds=False):
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
                model families. Run evalml.list_model_families("binary") to see options. Change `binary`
                to `multiclass` or `regression` depending on the problem type. Note that if allowed_pipelines is provided,
                this parameter will be ignored.

            cv: cross-validation method to use. Defaults to StratifiedKFold.

            tuner_class: the tuner class to use. Defaults to scikit-optimize tuner

            start_iteration_callback (callable): function called before each pipeline training iteration.
                Passed two parameters: pipeline_class, parameters.

            add_result_callback (callable): function called after each pipeline training iteration.
                Passed two parameters: results, trained_pipeline.

            additional_objectives (list): Custom set of objectives to score on.
                Will override default objectives for problem type if not empty.

            random_state (int, np.random.RandomState): The random seed/state. Defaults to 0.

            n_jobs (int or None): Non-negative integer describing level of parallelism used for pipelines.
                None and 1 are equivalent. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.

            verbose (boolean): If True, turn verbosity on. Defaults to True
        """
        try:
            self.problem_type = handle_problem_types(problem_type)
        except ValueError:
            raise ValueError('choose one of (binary, multiclass, regression) as problem_type')

        self.tuner_class = tuner_class or SKOptTuner
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.verbose = verbose
        self.optimize_thresholds = optimize_thresholds
        if cv is None:
            if self.problem_type.value == 'regression':
                self.cv = KFold(n_splits=3, random_state=random_state)
            else:
                self.cv = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)
        else:
            self.cv = cv
        if objective == 'auto':
            objective = self._DEFAULT_OBJECTIVES[self.problem_type.value]
        self.objective = get_objective(objective)

        if additional_objectives is None:
            additional_objectives = get_objectives(self.problem_type)
            # if our main objective is part of default set of objectives for problem_type, remove it
            existing_main_objective = next((obj for obj in additional_objectives if obj.name == self.objective.name), None)
            if existing_main_objective is not None:
                additional_objectives.remove(existing_main_objective)
        else:
            additional_objectives = [get_objective(o) for o in additional_objectives]
        self.additional_objectives = additional_objectives

        if max_time is None or isinstance(max_time, (int, float)):
            self.max_time = max_time
        elif isinstance(max_time, str):
            self.max_time = convert_to_seconds(max_time)
        else:
            raise TypeError("max_time must be a float, int, or string. Received a {}.".format(type(max_time)))

        self.max_pipelines = max_pipelines
        if self.max_pipelines is None and self.max_time is None:
            self.max_pipelines = 5
            logger.info("Using default limit of max_pipelines=5.\n")

        if patience and (not isinstance(patience, int) or patience < 0):
            raise ValueError("patience value must be a positive integer. Received {} instead".format(patience))

        if tolerance and (tolerance > 1.0 or tolerance < 0.0):
            raise ValueError("tolerance value must be a float between 0.0 and 1.0 inclusive. Received {} instead".format(tolerance))

        self.patience = patience
        self.tolerance = tolerance or 0.0
        self.results = {
            'pipeline_results': {},
            'search_order': []
        }
        self.trained_pipelines = {}
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

    @property
    def data_check_results(self):
        return self._data_check_results

    def __str__(self):
        def _print_list(obj_list):
            lines = ['\t{}'.format(o.name) for o in obj_list]
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
            f"Cross Validation: {self.cv}\n"
            f"Tuner: {self.tuner_class.__name__}\n"
            f"Start Iteration Callback: {_get_funct_name(self.start_iteration_callback)}\n"
            f"Add Result Callback: {_get_funct_name(self.add_result_callback)}\n"
            f"Additional Objectives: {_print_list(self.additional_objectives or [])}\n"
            f"Random State: {self.random_state}\n"
            f"n_jobs: {self.n_jobs}\n"
            f"Verbose: {self.verbose}\n"
            f"Optimize Thresholds: {self.optimize_thresholds}\n"
        )

        try:
            rankings_str = self.rankings.drop(['parameters'], axis='columns').to_string()
            rankings_desc = f"\nSearch Results: \n{'='*20}\n{rankings_str}"
        except KeyError:
            rankings_desc = ""

        return search_desc + rankings_desc

    def search(self, X, y, data_checks=None, feature_types=None, raise_errors=True, show_iteration_plot=True):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types, either numerical or categorical.
                Categorical features will automatically be encoded

            raise_errors (boolean): If True, raise errors and exit search if a pipeline errors during fitting. If False, set scores for the errored pipeline to NaN and continue search. Defaults to True.

            show_iteration_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.

            data_checks (DataChecks, None): A collection of data checks to run before searching for the best classifier. If data checks produce any errors, an exception will be thrown before the search begins. If None, uses DefaultDataChecks. Defaults to None.

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

        if data_checks is None:
            data_checks = DefaultDataChecks()

        if not isinstance(data_checks, DataChecks):
            raise ValueError("data_checks parameter must be a DataChecks object!")

        data_check_results = data_checks.validate(X, y)
        if len(data_check_results) > 0:
            self._data_check_results = data_check_results
            for message in self._data_check_results:
                if message.message_type == DataCheckMessageType.WARNING:
                    logger.warning(message)
                elif message.message_type == DataCheckMessageType.ERROR:
                    logger.error(message)
            if any([message.message_type == DataCheckMessageType.ERROR for message in self._data_check_results]):
                raise ValueError("Data checks raised some warnings and/or errors. Please see `self.data_check_results` for more information or pass data_checks=EmptyDataChecks() to search() to disable data checking.")

        if self.allowed_pipelines is None:
            logger.info("Generating pipelines to search over...")
            allowed_estimators = get_estimators(self.problem_type, self.allowed_model_families)
            logger.debug(f"allowed_estimators set to {[estimator.name for estimator in allowed_estimators]}")
            self.allowed_pipelines = [make_pipeline(X, y, estimator, self.problem_type) for estimator in allowed_estimators]

        if self.allowed_pipelines == []:
            raise ValueError("No allowed pipelines to search")

        self.allowed_model_families = list(set([p.model_family for p in (self.allowed_pipelines)]))

        logger.debug(f"allowed_pipelines set to {[pipeline.name for pipeline in self.allowed_pipelines]}")
        logger.debug(f"allowed_model_families set to {self.allowed_model_families}")

        self._validate_problem_type()
        self._automl_algorithm = IterativeAlgorithm(
            max_pipelines=self.max_pipelines,
            allowed_pipelines=self.allowed_pipelines,
            tuner_class=self.tuner_class,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            number_features=X.shape[1]
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

        if self.max_pipelines is None:
            pbar = tqdm(total=self.max_time, disable=not self.verbose, file=stdout, bar_format='{desc} |    Elapsed:{elapsed}')
        else:
            pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout, bar_format='{desc}   {percentage:3.0f}%|{bar}| Elapsed:{elapsed}')
        pbar._instances.clear()

        start = time.time()
        self._add_baseline_pipelines(X, y, pbar, raise_errors=raise_errors)

        current_batch_pipelines = []
        while self._check_stopping_condition(start):
            if len(current_batch_pipelines) == 0:
                try:
                    current_batch_pipelines = self._automl_algorithm.next_batch()
                except StopIteration:
                    logger.info('AutoML Algorithm out of recommendations, ending')
                    break
            pipeline = current_batch_pipelines.pop(0)
            parameters = pipeline.parameters
            logger.debug('Evaluating pipeline {}'.format(pipeline.name))
            logger.debug('Pipeline parameters: {}'.format(parameters))
            pbar.update(1)
            if self.start_iteration_callback:
                self.start_iteration_callback(pipeline.__class__, parameters)
            desc = "▹ {}: ".format(pipeline.name)
            if len(desc) > self._MAX_NAME_LEN:
                desc = desc[:self._MAX_NAME_LEN - 3] + "..."
            desc = desc.ljust(self._MAX_NAME_LEN)
            pbar.set_description_str(desc=desc, refresh=True)

            evaluation_results = self._evaluate(pipeline, X, y, raise_errors=raise_errors, pbar=pbar)
            score = evaluation_results['cv_score_mean']
            score_to_minimize = -score if self.objective.greater_is_better else score
            self._automl_algorithm.add_result(score_to_minimize, pipeline)

            desc = "✔" + desc[1:]
            pbar.set_description_str(desc=desc, refresh=True)
            if self.verbose:  # To force new line between progress bar iterations
                print('')

            if search_iteration_plot:
                search_iteration_plot.update()

        desc = "✔ Optimization finished"
        desc = desc.ljust(self._MAX_NAME_LEN)
        pbar.set_description_str(desc=desc, refresh=True)
        pbar.close()

    def _check_stopping_condition(self, start):
        should_continue = True
        num_pipelines = len(self.results['pipeline_results'])
        if num_pipelines == 0:
            return True

        # check max_time and max_pipelines
        elapsed = time.time() - start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_pipelines and num_pipelines >= self.max_pipelines:
            return False

        # check for early stopping
        if self.patience is None:
            return True

        first_id = self.results['search_order'][0]
        best_score = self.results['pipeline_results'][first_id]['score']
        num_without_improvement = 0
        for id in self.results['search_order'][1:]:
            curr_score = self.results['pipeline_results'][id]['score']
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
        if self.objective.problem_type != self.problem_type:
            raise ValueError("Given objective {} is not compatible with a {} problem.".format(self.objective.name, self.problem_type.value))
        for obj in self.additional_objectives:
            if obj.problem_type != self.problem_type:
                raise ValueError("Additional objective {} is not compatible with a {} problem.".format(obj.name, self.problem_type.value))

        for pipeline in self.allowed_pipelines:
            if not pipeline.problem_type == self.problem_type:
                raise ValueError("Given pipeline {} is not compatible with problem_type {}.".format(pipeline.name, self.problem_type.value))

    def _add_baseline_pipelines(self, X, y, pbar, raise_errors=True):
        if self.problem_type == ProblemTypes.BINARY:
            strategy_dict = {"strategy": "random_weighted"}
            baseline = ModeBaselineBinaryPipeline(parameters={"Baseline Classifier": strategy_dict})
        elif self.problem_type == ProblemTypes.MULTICLASS:
            strategy_dict = {"strategy": "random_weighted"}
            baseline = ModeBaselineMulticlassPipeline(parameters={"Baseline Classifier": strategy_dict})
        elif self.problem_type == ProblemTypes.REGRESSION:
            strategy_dict = {"strategy": "mean"}
            baseline = MeanBaselineRegressionPipeline(parameters={"Baseline Regressor": strategy_dict})

        if self.start_iteration_callback:
            self.start_iteration_callback(baseline.__class__, baseline.parameters)

        desc = "▹ {}: ".format(baseline.name)
        if len(desc) > self._MAX_NAME_LEN:
            desc = desc[:self._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self._MAX_NAME_LEN)
        pbar.set_description_str(desc=desc, refresh=True)

        baseline_results = self._compute_cv_scores(baseline, X, y, raise_errors=raise_errors, pbar=pbar)
        self._add_result(trained_pipeline=baseline,
                         parameters=strategy_dict,
                         training_time=baseline_results['training_time'],
                         cv_data=baseline_results['cv_data'],
                         cv_scores=baseline_results['cv_scores'])
        desc = "✔" + desc[1:]
        pbar.set_description_str(desc=desc, refresh=True)
        if self.verbose:  # To force new line between progress bar iterations
            print('')

    def _compute_cv_scores(self, pipeline, X, y, raise_errors=True, pbar=None):
        start = time.time()
        cv_data = []

        for train, test in self.cv.split(X, y):
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train], X.iloc[test]
            else:
                X_train, X_test = X[train], X[test]
            if isinstance(y, pd.Series):
                y_train, y_test = y.iloc[train], y.iloc[test]
            else:
                y_train, y_test = y[train], y[test]

            objectives_to_score = [self.objective] + self.additional_objectives
            try:
                X_threshold_tuning = None
                y_threshold_tuning = None

                if self.optimize_thresholds and self.objective.problem_type == ProblemTypes.BINARY and self.objective.can_optimize_threshold:
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=self.random_state)
                pipeline.fit(X_train, y_train)
                if self.objective.problem_type == ProblemTypes.BINARY:
                    pipeline.threshold = 0.5
                    if self.optimize_thresholds and self.objective.can_optimize_threshold:
                        y_predict_proba = pipeline.predict_proba(X_threshold_tuning)
                        if isinstance(y_predict_proba, pd.DataFrame):
                            y_predict_proba = y_predict_proba.iloc[:, 1]
                        else:
                            y_predict_proba = y_predict_proba[:, 1]
                        pipeline.threshold = self.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                scores = pipeline.score(X_test, y_test, objectives=objectives_to_score)
                score = scores[self.objective.name]
            except Exception as e:
                logger.error("Exception during automl search: {}".format(str(e)))
                if raise_errors:
                    raise e
                if pbar:
                    pbar.write(str(e))
                score = np.nan
                scores = OrderedDict(zip([n.name for n in self.additional_objectives], [np.nan] * len(self.additional_objectives)))
            ordered_scores = OrderedDict()
            ordered_scores.update({self.objective.name: score})
            ordered_scores.update(scores)
            ordered_scores.update({"# Training": len(y_train)})
            ordered_scores.update({"# Testing": len(y_test)})
            cv_data.append({"all_objective_scores": ordered_scores, "score": score})

        training_time = time.time() - start
        cv_scores = pd.Series([fold['score'] for fold in cv_data])
        return {'cv_data': cv_data, 'training_time': training_time, 'cv_scores': cv_scores, 'cv_score_mean': cv_scores.mean()}

    def _add_result(self, trained_pipeline, parameters, training_time, cv_data, cv_scores):
        cv_score = cv_scores.mean()
        # calculate high_variance_cv
        # if the coefficient of variance is greater than .2
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            high_variance_cv = (cv_scores.std() / cv_scores.mean()) > .2

        pipeline_name = trained_pipeline.name
        pipeline_summary = trained_pipeline.summary
        pipeline_id = len(self.results['pipeline_results'])

        self.results['pipeline_results'][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline_name,
            "pipeline_class": type(trained_pipeline),
            "pipeline_summary": pipeline_summary,
            "parameters": parameters,
            "score": cv_score,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data
        }

        self.results['search_order'].append(pipeline_id)

        if self.add_result_callback:
            self.add_result_callback(self.results['pipeline_results'][pipeline_id], trained_pipeline)

        self._save_pipeline(pipeline_id, trained_pipeline)

    def _evaluate(self, pipeline, X, y, raise_errors=True, pbar=None):
        parameters = pipeline.parameters
        evaluation_results = self._compute_cv_scores(pipeline, X, y, raise_errors=raise_errors, pbar=pbar)
        logger.debug('Adding results for pipeline {}\nparameters {}\nevaluation_results {}'.format(pipeline.name, parameters, evaluation_results))

        self._add_result(trained_pipeline=pipeline,
                         parameters=parameters,
                         training_time=evaluation_results['training_time'],
                         cv_data=evaluation_results['cv_data'],
                         cv_scores=evaluation_results['cv_scores'])

        logger.debug('Adding results complete')
        return evaluation_results

    def _save_pipeline(self, pipeline_id, trained_pipeline):
        self.trained_pipelines[pipeline_id] = trained_pipeline

    def get_pipeline(self, pipeline_id):
        """Retrieves trained pipeline

        Arguments:
            pipeline_id (int): pipeline to retrieve

        Returns:
            Pipeline: pipeline associated with id
        """
        if pipeline_id not in self.trained_pipelines:
            raise RuntimeError("Pipeline not found")

        return self.trained_pipelines[pipeline_id]

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
        if pipeline_id not in self.results['pipeline_results']:
            raise RuntimeError("Pipeline not found")

        pipeline = self.get_pipeline(pipeline_id)
        pipeline_results = self.results['pipeline_results'][pipeline_id]

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
        """Fits and evaluates a given pipeline then adds the results to the AutoML rankings. Please use the same data as previous runs of AutoML search.
        If pipeline already exists in rankings this method will return `None`.

        Arguments:
            pipeline (PipelineBase): pipeline to train and evaluate.

            X (pd.DataFrame): the input training data of shape [n_samples, n_features].

            y (pd.Series): the target training labels of length [n_samples].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        pipeline_rows = self.full_rankings[self.full_rankings['pipeline_name'] == pipeline.name]
        for parameter in pipeline_rows['parameters']:
            if pipeline.parameters == parameter:
                return
        self._evaluate(pipeline, X, y, raise_errors=True)

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

        rankings_df = pd.DataFrame(self.results['pipeline_results'].values())
        rankings_df = rankings_df[["id", "pipeline_name", "score", "high_variance_cv", "parameters"]]
        rankings_df.sort_values("score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)
        return rankings_df

    @property
    def best_pipeline(self):
        """Returns the best model found"""
        best = self.rankings.iloc[0]
        return self.get_pipeline(best["id"])

    def save(self, file_path):
        """Saves AutoML object at file path

        Arguments:
            file_path (str) : location to save file

        Returns:
            None
        """
        with open(file_path, 'wb') as f:
            cloudpickle.dump(self, f)

    @staticmethod
    def load(file_path):
        """Loads AutoML object at file path

        Arguments:
            file_path (str) : location to find file to load

        Returns:
            AutoSearchBase object
        """
        with open(file_path, 'rb') as f:
            return cloudpickle.load(f)
