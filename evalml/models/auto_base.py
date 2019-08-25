import random
import time
from sys import stdout

import numpy as np
import pandas as pd
from colorama import Style
from tqdm import tqdm

from evalml.objectives import get_objective
from evalml.pipelines import get_pipelines
from evalml.tuners import SKOptTuner


class AutoBase:
    def __init__(self, problem_type, tuner, cv, objective, max_pipelines, max_time,
                 model_types, default_objectives, random_state, verbose):

        if tuner is None:
            tuner = SKOptTuner

        self.objective = get_objective(objective)

        self.max_pipelines = max_pipelines
        self.max_time = max_time
        self.model_types = model_types
        self.cv = cv
        self.verbose = verbose

        self.possible_pipelines = get_pipelines(problem_type=problem_type, model_types=model_types)

        self.results = {}
        self.trained_pipelines = {}
        self.random_state = random_state

        self.possible_model_types = list(set([p.model_type for p in self.possible_pipelines]))

        self.tuners = {}
        self.search_spaces = {}
        for p in self.possible_pipelines:
            space = list(p.hyperparameters.items())
            self.tuners[p.name] = tuner([s[1] for s in space], random_state=random_state)
            self.search_spaces[p.name] = [s[0] for s in space]

        self.default_objectives = default_objectives

    def _log(self, msg, color=None, new_line=True):
        if color:
            msg = color + msg + Style.RESET_ALL

        if new_line:
            print(msg)
        else:
            print(msg, end="")

    def _log_title(self, title):
        self._log("*" * (len(title) + 4), color=Style.BRIGHT)
        self._log("* %s *" % title, color=Style.BRIGHT)
        self._log("*" * (len(title) + 4), color=Style.BRIGHT)
        self._log("")

    def _log_subtitle(self, title, underline="=", color=None):
        self._log("%s" % title, color=color)
        self._log(underline * len(title), color=color)

    def fit(self, X, y, feature_types=None, raise_errors=False):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

            raise_errors (boolean): If true, raise errors and exit search if a pipeline errors during fitting

        Returns:

            self
        """
        if self.verbose:
            self._log_title("Beginning pipeline search")
            self._log("Optimizing for %s. " % self.objective.name, new_line=False)

            if self.objective.greater_is_better:
                self._log("Greater score is better.\n")
            else:
                self._log("Lower score is better.\n")

            self._log("Searching up to %s pipelines. " % self.max_pipelines, new_line=False)
            if self.max_time:
                self._log("Will stop searching for new pipelines after %d seconds.\n" % self.max_time)
            else:
                self._log("No time limit is set. Set one using max_time parameter.\n")
            self._log("Possible model types: %s\n" % ", ".join(self.possible_model_types))

        pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout)

        start = time.time()
        for n in pbar:
            elapsed = time.time() - start
            if self.max_time and elapsed > self.max_time:
                self._log("\n\nMax time elapsed. Stopping search early.")
                break
            self._do_iteration(X, y, pbar, raise_errors)

        pbar.close()

        self._log("\n✔ Optimization finished")

    def _do_iteration(self, X, y, pbar, raise_errors):
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

        pbar.set_description("Testing %s" % (pipeline_class.name))

        start = time.time()
        scores = []
        all_objective_scores = []
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
                score, other_scores = pipeline.score(X_test, y_test, other_objectives=self.default_objectives)

            except Exception as e:
                if raise_errors:
                    raise e
                pbar.write(str(e))
                score = np.nan
                other_scores = dict(zip([n.name for n in self.default_objectives], [np.nan] * len(self.default_objectives)))

            other_scores[self.objective.name] = score
            other_scores["# Training"] = len(y_train)
            other_scores["# Testing"] = len(y_test)

            scores.append(score)
            all_objective_scores.append(other_scores)

        training_time = time.time() - start

        # save the result and continue
        self._add_result(
            trained_pipeline=pipeline,
            parameters=parameters,
            scores=scores,
            all_objective_scores=all_objective_scores,
            training_time=training_time
        )

    def _select_pipeline(self):
        return random.choice(self.possible_pipelines)

    def _propose_parameters(self, pipeline_class):
        values = self.tuners[pipeline_class.name].propose()
        space = self.search_spaces[pipeline_class.name]
        proposal = zip(space, values)
        return list(proposal)

    def _add_result(self, trained_pipeline, parameters, scores, all_objective_scores, training_time):
        score = pd.Series(scores).mean()

        if self.objective.greater_is_better:
            score_to_minimize = -score
        else:
            score_to_minimize = score

        self.tuners[trained_pipeline.name].add([p[1] for p in parameters], score_to_minimize)

        # calculate high_variance_cv
        # if the coefficient of variance is greater than .2
        s = pd.Series(scores)
        high_variance_cv = (s.std() / s.mean()) > .2

        pipeline_name = trained_pipeline.__class__.__name__
        pipeline_id = len(self.results)

        self.results[pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline_name,
            "parameters": dict(parameters),
            "score": score,
            "high_variance_cv": high_variance_cv,
            "scores": scores,
            "all_objective_scores": all_objective_scores,
            "training_time": training_time,
        }

        self._save_pipeline(pipeline_id, trained_pipeline)

    def _save_pipeline(self, pipeline_id, trained_pipeline):
        self.trained_pipelines[pipeline_id] = trained_pipeline

    def get_pipeline(self, pipeline_id):
        if pipeline_id not in self.trained_pipelines:
            raise RuntimeError("Pipeline not found")

        return self.trained_pipelines[pipeline_id]

    def describe_pipeline(self, pipeline_id, return_dict=False):
        """Describe a pipeline

        Arguments:
            pipeline_id (int): pipeline to describe
            return_dict (bool): If True, return dictionary of information
                about pipeline. Defaults to false

        Returns:
            description
        """
        if pipeline_id not in self.results:
            raise RuntimeError("Pipeline not found")

        pipeline = self.get_pipeline(pipeline_id)
        pipeline_results = self.results[pipeline_id]

        self._log_title("Pipeline Description")

        better_string = "lower is better"
        if pipeline.objective.greater_is_better:
            better_string = "greater is better"

        self._log("Pipeline Name: %s" % pipeline.name)
        self._log("Model type: %s" % pipeline.model_type)
        self._log("Objective: %s (%s)" % (pipeline.objective.name, better_string))
        self._log("Total training time (including CV): %.1f seconds\n" % pipeline_results["training_time"])

        self._log_subtitle("Parameters")
        for item in pipeline_results["parameters"].items():
            self._log("• %s: %s" % item)

        self._log_subtitle("\nCross Validation")

        if pipeline_results["high_variance_cv"]:
            self._log("Warning! High variance within cross validation scores. " +
                      "Model may not perform as estimated on unseen data.")

        all_objective_scores = pd.DataFrame(pipeline_results["all_objective_scores"])

        for c in all_objective_scores:
            if c in ["# Training", "# Testing"]:
                all_objective_scores[c] = all_objective_scores[c].astype("object")

            mean = all_objective_scores[c].mean(axis=0)
            std = all_objective_scores[c].std(axis=0)
            all_objective_scores.loc["mean", c] = mean
            all_objective_scores.loc["std", c] = std
            all_objective_scores.loc["coef of var", c] = std / mean

        all_objective_scores = all_objective_scores.fillna("-")


        with pd.option_context('display.float_format', '{:.3f}'.format, 'expand_frame_repr', False):
            self._log(all_objective_scores)

        if return_dict:
            return pipeline_results

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
        ascending = True
        if self.objective.greater_is_better:
            ascending = False

        rankings_df = pd.DataFrame(self.results.values())
        rankings_df = rankings_df[["id", "pipeline_name", "score", "high_variance_cv", "parameters"]]
        rankings_df.sort_values("score", ascending=ascending, inplace=True)
        rankings_df.reset_index(drop=True, inplace=True)

        return rankings_df

    @property
    def best_pipeline(self):
        """Returns the best model found"""
        best = self.rankings.iloc[0]
        return self.get_pipeline(best["id"])
