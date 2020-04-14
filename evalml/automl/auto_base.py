import inspect
import time
from collections import OrderedDict
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .pipeline_search_plots import PipelineSearchPlots

from evalml import guardrails
from evalml.objectives import get_objective, get_objectives
from evalml.pipelines import get_pipelines
from evalml.pipelines.components import handle_component
from evalml.problem_types import ProblemTypes
from evalml.tuners import SKOptTuner
from evalml.utils import Logger, convert_to_seconds, get_random_state

logger = Logger()


class AutoBase:

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    def __init__(self, problem_type, tuner, cv, objective, max_pipelines, max_time,
                 patience, tolerance, allowed_model_families, detect_label_leakage, start_iteration_callback,
                 add_result_callback, additional_objectives, random_state, n_jobs, verbose, optimize_thresholds=False):
        if tuner is None:
            tuner = SKOptTuner
        self.problem_type = problem_type
        self.max_pipelines = max_pipelines
        self.allowed_model_families = allowed_model_families
        self.detect_label_leakage = detect_label_leakage
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.cv = cv
        self.verbose = verbose
        self.optimize_thresholds = optimize_thresholds
        self.possible_pipelines = get_pipelines(problem_type=self.problem_type, model_families=allowed_model_families)
        self.objective = get_objective(objective)
        if self.problem_type != self.objective.problem_type:
            raise ValueError("Given objective {} is not compatible with a {} problem.".format(self.objective.name, self.problem_type.value))

        logger.verbose = verbose

        if additional_objectives is not None:
            additional_objectives = [get_objective(o) for o in additional_objectives]
        else:
            additional_objectives = get_objectives(self.problem_type)

            # if our main objective is part of default set of objectives for problem_type, remove it
            existing_main_objective = next((obj for obj in additional_objectives if obj.name == self.objective.name), None)
            if existing_main_objective is not None:
                additional_objectives.remove(existing_main_objective)

        self.additional_objectives = additional_objectives

        if max_time is None or isinstance(max_time, (int, float)):
            self.max_time = max_time
        elif isinstance(max_time, str):
            self.max_time = convert_to_seconds(max_time)
        else:
            raise TypeError("max_time must be a float, int, or string. Received a {}.".format(type(max_time)))

        if patience and (not isinstance(patience, int) or patience < 0):
            raise ValueError("patience value must be a positive integer. Received {} instead".format(patience))

        if tolerance and (tolerance > 1.0 or tolerance < 0.0):
            raise ValueError("tolerance value must be a float between 0.0 and 1.0 inclusive. Received {} instead".format(tolerance))

        self.patience = patience
        self.tolerance = tolerance if tolerance else 0.0
        self.results = {
            'pipeline_results': {},
            'search_order': []
        }
        self.trained_pipelines = {}
        self.random_state = get_random_state(random_state)

        self.n_jobs = n_jobs
        self.possible_model_families = list(set([p.model_family for p in self.possible_pipelines]))

        self.tuners = {}
        self.search_spaces = {}
        for p in self.possible_pipelines:
            space = list(p.hyperparameters.items())
            self.tuners[p.name] = tuner([s[1] for s in space], random_state=self.random_state)
            self.search_spaces[p.name] = [s[0] for s in space]
        self._MAX_NAME_LEN = 40
        self._next_pipeline_class = None
        self.plot_metrics = []
        try:
            self.plot = PipelineSearchPlots(self)

        except ImportError:
            logger.log("Warning: unable to import plotly; skipping pipeline search plotting\n")
            self.plot = None

    def search(self, X, y, feature_types=None, raise_errors=False, show_iteration_plot=True):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types, either numerical or categorical.
                Categorical features will automatically be encoded

            raise_errors (boolean): If true, raise errors and exit search if a pipeline errors during fitting

            show_iteration_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.

        Returns:

            self
        """
        # don't show iteration plot outside of a jupyter notebook
        if show_iteration_plot is True:
            try:
                get_ipython
            except NameError:
                show_iteration_plot = False

        # make everything pandas objects
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        if self.problem_type != ProblemTypes.REGRESSION:
            self._check_multiclass(y)

        logger.log_title("Beginning pipeline search")
        logger.log("Optimizing for %s. " % self.objective.name, new_line=False)

        if self.objective.greater_is_better:
            logger.log("Greater score is better.\n")
        else:
            logger.log("Lower score is better.\n")

        # Set default max_pipeline if none specified
        if self.max_pipelines is None and self.max_time is None:
            self.max_pipelines = 5
            logger.log("No search limit is set. Set using max_time or max_pipelines.\n")

        if self.max_pipelines:
            logger.log("Searching up to %s pipelines. " % self.max_pipelines)
        if self.max_time:
            logger.log("Will stop searching for new pipelines after %d seconds.\n" % self.max_time)
            logger.log("Possible model families: %s\n" % ", ".join([model.value for model in self.possible_model_families]))

        if self.detect_label_leakage:
            leaked = guardrails.detect_label_leakage(X, y)
            if len(leaked) > 0:
                leaked = [str(k) for k in leaked.keys()]
                logger.log("WARNING: Possible label leakage: %s" % ", ".join(leaked))

        search_iteration_plot = None
        if self.plot:
            search_iteration_plot = self.plot.search_iteration_plot(interactive_plot=show_iteration_plot)

        if self.max_pipelines is None:
            pbar = tqdm(total=self.max_time, disable=not self.verbose, file=stdout, bar_format='{desc} |    Elapsed:{elapsed}')
            pbar._instances.clear()
        else:
            pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout, bar_format='{desc}   {percentage:3.0f}%|{bar}| Elapsed:{elapsed}')
            pbar._instances.clear()

        start = time.time()
        while self._check_stopping_condition(start):
            self._do_iteration(X, y, pbar, raise_errors)
            if search_iteration_plot:
                search_iteration_plot.update()
        desc = "✔ Optimization finished"
        desc = desc.ljust(self._MAX_NAME_LEN)
        pbar.set_description_str(desc=desc, refresh=True)
        pbar.close()

    def _check_stopping_condition(self, start):
        # get new pipeline and check tuner
        self._next_pipeline_class = self._select_pipeline()
        if self.tuners[self._next_pipeline_class.name].is_search_space_exhausted():
            return False

        should_continue = True
        num_pipelines = len(self.results['pipeline_results'])
        if num_pipelines == 0:
            return True

        # check max_time and max_pipelines
        elapsed = time.time() - start
        if self.max_time and elapsed >= self.max_time:
            return False
        elif self.max_pipelines:
            if num_pipelines >= self.max_pipelines:
                return False
            elif self.max_time and elapsed >= self.max_time:
                logger.log("\n\nMax time elapsed. Stopping search early.")
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
                logger.log("\n\n{} iterations without improvement. Stopping search early...".format(self.patience))
                return False
        return should_continue

    def _check_multiclass(self, y):
        if y.nunique() <= 2:
            return
        if self.objective.problem_type != ProblemTypes.MULTICLASS:
            raise ValueError("Given objective {} is not compatible with a multiclass problem.".format(self.objective.name))
        for obj in self.additional_objectives:
            if obj.problem_type != ProblemTypes.MULTICLASS:
                raise ValueError("Additional objective {} is not compatible with a multiclass problem.".format(obj.name))

    def _transform_parameters(self, pipeline_class, parameters, number_features):
        new_parameters = {}
        component_graph = [handle_component(c) for c in pipeline_class.component_graph]
        for component in component_graph:
            component_parameters = {}
            component_class = component.__class__

            # Inspects each component and adds the following parameters when needed
            if 'n_jobs' in inspect.signature(component_class.__init__).parameters:
                component_parameters['n_jobs'] = self.n_jobs
            if 'number_features' in inspect.signature(component_class.__init__).parameters:
                component_parameters['number_features'] = number_features

            # Inspects each component and checks the parameters list for the right parameters
            # Sk_opt tuner returns a list of (name, value) tuples so must be accessed as follows
            for parameter in parameters:
                if parameter[0] in inspect.signature(component_class.__init__).parameters:
                    component_parameters[parameter[0]] = parameter[1]

            new_parameters[component.name] = component_parameters
        return new_parameters

    def _do_iteration(self, X, y, pbar, raise_errors):
        pbar.update(1)

        # propose the next best parameters for this piepline
        parameters = self._propose_parameters(self._next_pipeline_class)

        # fit an score the pipeline
        pipeline = self._next_pipeline_class(parameters=self._transform_parameters(self._next_pipeline_class, parameters, X.shape[1]))

        if self.start_iteration_callback:
            self.start_iteration_callback(self._next_pipeline_class, parameters)

        desc = "▹ {}: ".format(self._next_pipeline_class.name)
        if len(desc) > self._MAX_NAME_LEN:
            desc = desc[:self._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self._MAX_NAME_LEN)
        pbar.set_description_str(desc=desc, refresh=True)

        start = time.time()
        cv_data = []
        plot_data = []

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
                    X_train, X_threshold_tuning, y_train, y_threshold_tuning = train_test_split(X_train, y_train, test_size=0.2, random_state=pipeline.estimator.random_state)
                pipeline.fit(X_train, y_train)
                if self.objective.problem_type == ProblemTypes.BINARY:
                    pipeline.threshold = 0.5
                    if self.optimize_thresholds and self.objective.can_optimize_threshold:
                        y_predict_proba = pipeline.predict_proba(X_threshold_tuning)
                        y_predict_proba = y_predict_proba[:, 1]
                        pipeline.threshold = self.objective.optimize_threshold(y_predict_proba, y_threshold_tuning, X=X_threshold_tuning)
                scores = pipeline.score(X_test, y_test, objectives=objectives_to_score)
                score = scores[self.objective.name]
            except Exception as e:
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

        # save the result and continue
        self._add_result(trained_pipeline=pipeline,
                         parameters=parameters,
                         training_time=training_time,
                         cv_data=cv_data,
                         plot_data=plot_data)

        desc = "✔" + desc[1:]
        pbar.set_description_str(desc=desc, refresh=True)
        if self.verbose:  # To force new line between progress bar iterations
            print('')

    def _select_pipeline(self):
        return self.random_state.choice(self.possible_pipelines)

    def _propose_parameters(self, pipeline_class):
        values = self.tuners[pipeline_class.name].propose()
        space = self.search_spaces[pipeline_class.name]
        proposal = zip(space, values)
        return list(proposal)

    def _add_result(self, trained_pipeline, parameters, training_time, cv_data, plot_data):
        scores = pd.Series([fold["score"] for fold in cv_data])
        score = scores.mean()

        if self.objective.greater_is_better:
            score_to_minimize = -score
        else:
            score_to_minimize = score

        self.tuners[trained_pipeline.name].add([p[1] for p in parameters], score_to_minimize)
        # calculate high_variance_cv
        # if the coefficient of variance is greater than .2
        high_variance_cv = (scores.std() / scores.mean()) > .2

        pipeline_name = trained_pipeline.name
        pipeline_summary = trained_pipeline.summary
        pipeline_id = len(self.results['pipeline_results'])

        self.results['pipeline_results'][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline_name,
            "pipeline_summary": pipeline_summary,
            "parameters": dict(parameters),
            "score": score,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data,
            "plot_data": plot_data
        }

        self.results['search_order'].append(pipeline_id)

        if self.add_result_callback:
            self.add_result_callback(self.results['pipeline_results'][pipeline_id], trained_pipeline)

        self._save_pipeline(pipeline_id, trained_pipeline)

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
        logger.log_subtitle("Training")
        # Ideally, we want this information available on pipeline instead
        logger.log("Training for {} problems.".format(self.problem_type))
        logger.log("Total training time (including CV): %.1f seconds" % pipeline_results["training_time"])
        logger.log_subtitle("Cross Validation", underline="-")

        if pipeline_results["high_variance_cv"]:
            logger.log("Warning! High variance within cross validation scores. " +
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
            all_objective_scores.loc["coef of var", c] = std / mean

        all_objective_scores = all_objective_scores.fillna("-")

        with pd.option_context('display.float_format', '{:.3f}'.format, 'expand_frame_repr', False):
            logger.log(all_objective_scores)

        if return_dict:
            return pipeline_results

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
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
