import random
import time
from collections import OrderedDict
from sys import stdout
from threading import Event, Thread

import numpy as np
import pandas as pd
from tqdm import tqdm

from .pipeline_search_plots import PipelineSearchPlots

from evalml import guardrails
from evalml.objectives import get_objective, get_objectives
from evalml.pipelines import get_pipelines
from evalml.problem_types import ProblemTypes
from evalml.tuners import SKOptTuner
from evalml.utils import Logger, convert_to_seconds, sigint_handler


class AutoBase:

    # Necessary for "Plotting" documentation, since Sphinx does not work well with instance attributes.
    plot = PipelineSearchPlots

    def __init__(self, problem_type, tuner, cv, objective, max_pipelines, max_time,
                 patience, tolerance, model_types, detect_label_leakage, start_iteration_callback,
                 add_result_callback, additional_objectives, random_state, verbose):
        if tuner is None:
            tuner = SKOptTuner
        self.objective = get_objective(objective)
        self.problem_type = problem_type
        self.max_pipelines = max_pipelines
        self.model_types = model_types
        self.detect_label_leakage = detect_label_leakage
        self.start_iteration_callback = start_iteration_callback
        self.add_result_callback = add_result_callback
        self.cv = cv
        self.verbose = verbose
        self.logger = Logger(self.verbose)
        self.possible_pipelines = get_pipelines(problem_type=self.problem_type, model_types=model_types)
        self.objective = get_objective(objective)
        if self.problem_type not in self.objective.problem_types:
            raise ValueError("Given objective {} is not compatible with a {} problem.".format(self.objective.name, self.problem_type.value))

        if additional_objectives is not None:
            additional_objectives = [get_objective(o) for o in additional_objectives]
        else:
            additional_objectives = get_objectives(self.problem_type)

            # if our main objective is part of default set of objectives for problem_type, remove it
            existing_main_objective = next((obj for obj in additional_objectives if obj.name == self.objective.name), None)
            if existing_main_objective is not None:
                additional_objectives.remove(existing_main_objective)

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
        self.random_state = random_state
        random.seed(self.random_state)
        np.random.seed(seed=self.random_state)
        self.possible_model_types = list(set([p.model_type for p in self.possible_pipelines]))

        self.tuners = {}
        self.search_spaces = {}
        for p in self.possible_pipelines:
            space = list(p.hyperparameters.items())
            self.tuners[p.name] = tuner([s[1] for s in space], random_state=random_state)
            self.search_spaces[p.name] = [s[0] for s in space]

        self.additional_objectives = additional_objectives
        self._MAX_NAME_LEN = 40

        self.plot = PipelineSearchPlots(self)

    def search(self, X, y, feature_types=None, raise_errors=False, show_iteration_plot=True):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

            raise_errors (boolean): If true, raise errors and exit search if a pipeline errors during fitting

            show_iteration_plot (boolean, True): Shows an iteration vs. score plot in Jupyter notebook.
                Disabled by default in non-Jupyter enviroments.

        Returns:

            self
        """
        def search_loop(stop_event, self, start, X, y, pbar, raise_errors, plot, desc):
            while not stop_event.is_set() and self._check_stopping_condition(start):
                pipeline_thread = Thread(target=self._do_iteration, args=[X, y, pbar, raise_errors], daemon=True)
                pipeline_thread.start()
                pipeline_thread.join()
                plot.update()

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

        self.logger.log_title("Beginning pipeline search")
        self.logger.log("Optimizing for %s. " % self.objective.name, new_line=False)

        if self.objective.greater_is_better:
            self.logger.log("Greater score is better.\n")
        else:
            self.logger.log("Lower score is better.\n")

        # Set default max_pipeline if none specified
        if self.max_pipelines is None and self.max_time is None:
            self.max_pipelines = 5
            self.logger.log("No search limit is set. Set using max_time or max_pipelines.\n")

        if self.max_pipelines:
            self.logger.log("Searching up to %s pipelines. " % self.max_pipelines)
        if self.max_time:
            self.logger.log("Will stop searching for new pipelines after %d seconds.\n" % self.max_time)
        self.logger.log("Possible model types: %s\n" % ", ".join([model.value for model in self.possible_model_types]))

        if self.detect_label_leakage:
            leaked = guardrails.detect_label_leakage(X, y)
            if len(leaked) > 0:
                leaked = [str(k) for k in leaked.keys()]
                self.logger.log("WARNING: Possible label leakage: %s" % ", ".join(leaked))

        plot = self.plot.search_iteration_plot(interactive_plot=show_iteration_plot)

        if self.max_pipelines is None:
            pbar = tqdm(total=self.max_time, disable=not self.verbose, file=stdout, bar_format='{desc} |    Elapsed:{elapsed}')
            pbar._instances.clear()
        else:
            pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout, bar_format='{desc}   {percentage:3.0f}%|{bar}| Elapsed:{elapsed}')
            pbar._instances.clear()

        start = 0
        desc = ''
        stop_event = Event()
        search_thread = Thread(target=search_loop, args=[stop_event, self, start, X, y, pbar, raise_errors, plot, desc], daemon=True)
        sigint_handler.setup_signal(search_thread, stop_event)
        start = time.time()
        search_thread.start()
        search_thread.join()
        # sigint_handler.remove_signal()
        if stop_event.is_set():
            desc = u"Optimization terminated early"
        else:
            desc = u"✔ Optimization finished"
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
        elif self.max_pipelines:
            if num_pipelines >= self.max_pipelines:
                return False
            elif self.max_time and elapsed >= self.max_time:
                self.logger.log("\n\nMax time elapsed. Stopping search early.")
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
                self.logger.log("\n\n{} iterations without improvement. Stopping search early...".format(self.patience))
                return False
        return should_continue

    def _check_multiclass(self, y):
        if y.nunique() <= 2:
            return
        if ProblemTypes.MULTICLASS not in self.objective.problem_types:
            raise ValueError("Given objective {} is not compatible with a multiclass problem.".format(self.objective.name))
        for obj in self.additional_objectives:
            if ProblemTypes.MULTICLASS not in obj.problem_types:
                raise ValueError("Additional objective {} is not compatible with a multiclass problem.".format(obj.name))

    def _do_iteration(self, X, y, pbar, raise_errors):
        pbar.update(1)
        # determine which pipeline to build
        pipeline_class = self._select_pipeline()

        # propose the next best parameters for this piepline
        parameters = self._propose_parameters(pipeline_class)
        # fit an score the pipeline
        pipeline = pipeline_class(
            objective=self.objective,
            random_state=self.random_state,
            n_jobs=-1,
            number_features=X.shape[1],
            **dict(parameters)
        )

        if self.start_iteration_callback:
            self.start_iteration_callback(pipeline_class, parameters)

        desc = "▹ {}: ".format(pipeline_class.name)
        if len(desc) > self._MAX_NAME_LEN:
            desc = desc[:self._MAX_NAME_LEN - 3] + "..."
        desc = desc.ljust(self._MAX_NAME_LEN)
        pbar.set_description_str(desc=desc, refresh=True)

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

            try:
                pipeline.fit(X_train, y_train)
                score, other_scores = pipeline.score(X_test, y_test, other_objectives=self.additional_objectives)
            except Exception as e:
                if raise_errors:
                    raise e
                if pbar:
                    pbar.write(str(e))
                score = np.nan
                other_scores = OrderedDict(zip([n.name for n in self.additional_objectives], [np.nan] * len(self.additional_objectives)))
            ordered_scores = OrderedDict()
            ordered_scores.update({self.objective.name: score})
            ordered_scores.update(other_scores)
            ordered_scores.update({"# Training": len(y_train)})
            ordered_scores.update({"# Testing": len(y_test)})
            cv_data.append({"all_objective_scores": ordered_scores, "score": score})

        training_time = time.time() - start

        # save the result and continue
        self._add_result(trained_pipeline=pipeline,
                         parameters=parameters,
                         training_time=training_time,
                         cv_data=cv_data)

        desc = "✔" + desc[1:]
        pbar.set_description_str(desc=desc, refresh=True)
        if self.verbose:  # To force new line between progress bar iterations
            print('')

    def _select_pipeline(self):
        return random.choice(self.possible_pipelines)

    def _propose_parameters(self, pipeline_class):
        values = self.tuners[pipeline_class.name].propose()
        space = self.search_spaces[pipeline_class.name]
        proposal = zip(space, values)
        return list(proposal)

    def _add_result(self, trained_pipeline, parameters, training_time, cv_data):
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

        pipeline_class_name = trained_pipeline.__class__.__name__
        pipeline_name = trained_pipeline.name
        pipeline_id = len(self.results['pipeline_results'])

        self.results['pipeline_results'][pipeline_id] = {
            "id": pipeline_id,
            "pipeline_class_name": pipeline_class_name,
            "pipeline_name": pipeline_name,
            "parameters": dict(parameters),
            "score": score,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
            "cv_data": cv_data
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
        self.logger.log_subtitle("Training")
        # Ideally, we want this information available on pipeline instead
        self.logger.log("Training for {} problems.".format(self.problem_type))
        self.logger.log("Total training time (including CV): %.1f seconds" % pipeline_results["training_time"])
        self.logger.log_subtitle("Cross Validation", underline="-")

        if pipeline_results["high_variance_cv"]:
            self.logger.log("Warning! High variance within cross validation scores. " +
                            "Model may not perform as estimated on unseen data.")

        all_objective_scores = [fold["all_objective_scores"] for fold in pipeline_results["cv_data"]]
        all_objective_scores = pd.DataFrame(all_objective_scores)

        # note: we need to think about how to better handle metrics we don't want to display in our chart
        # currently, just dropping the columns before displaying
        all_objective_scores = all_objective_scores.drop(["ROC", "Confusion Matrix"], axis=1, errors="ignore")

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
            self.logger.log(all_objective_scores)

        if return_dict:
            return pipeline_results

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
        ascending = True
        if self.objective.greater_is_better:
            ascending = False

        rankings_df = pd.DataFrame(self.results['pipeline_results'].values())
        rankings_df = rankings_df[["id", "pipeline_class_name", "score", "high_variance_cv", "parameters"]]
        rankings_df.sort_values("score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)
        return rankings_df

    @property
    def best_pipeline(self):
        """Returns the best model found"""
        best = self.rankings.iloc[0]
        return self.get_pipeline(best["id"])
