# from evalml.pipelines import get_pipelines_by_model_type
import random
import time
from sys import stdout

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .auto_base import AutoBase

from evalml.objectives import get_objective, standard_metrics
from evalml.pipelines import get_pipelines
from evalml.preprocessing import split_data
from evalml.tuners import SKOptTuner


class AutoClassifier(AutoBase):
    def __init__(self,
                 objective=None,
                 max_pipelines=5,
                 max_time=60 * 5,
                 model_types=None,
                 cv=None,
                 tuner=None,
                 random_state=0,
                 verbose=True):
        """Automated classifier pipeline search

        Arguments:
            objective (Object): the objective to optimize
            max_pipelines (int): maximum number of pipelines to search
            max_time (int): maximum time in seconds to search for pipelines.
                won't start new pipeline search after this duration has elapsed
            model_types (list): The model types to search. By default searches over all
                model_types. Run evalml.list_model_types() to see options.
            cv: cross validation method to use. By default StratifiedKFold
            tuner: the tuner class to use. Defaults to scikit-optimize tuner
            random_state (int): the random_state
            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "precision"

        if tuner is None:
            tuner = SKOptTuner

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state)

        self.objective = get_objective(objective)
        self.max_pipelines = max_pipelines
        self.max_time = max_time
        self.model_types = model_types
        self.verbose = verbose
        self.results = {}
        self.trained_pipelines = {}
        self.random_state = random_state
        self.cv = cv
        self.possible_pipelines = get_pipelines(model_types=model_types)
        self.possible_model_types = list(set([p.model_type for p in self.possible_pipelines]))

        self.tuners = {}
        self.search_spaces = {}
        for p in self.possible_pipelines:
            space = list(p.hyperparameters.items())
            self.tuners[p.name] = tuner([s[1] for s in space], random_state=random_state)
            self.search_spaces[p.name] = space

        self.default_objectives = [
            standard_metrics.F1(),
            standard_metrics.Precision(),
            standard_metrics.Recall(),
            standard_metrics.AUC(),
            standard_metrics.LogLoss()
        ]

    def fit(self, X, y, feature_types=None):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

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

            self._log("Searching up to %s pipelines.\n" % self.max_pipelines)
            self._log("Possible model types: %s\n" % ", ".join(self.possible_model_types))

        pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout)

        start = time.time()
        for n in pbar:
            elapsed = time.time() - start
            if elapsed > self.max_time:
                self._log("\n\nMax time elapsed. Stopping search early.")
                break
            self._do_iteration(X, y, pbar)

        pbar.close()

        self._log("\n✔ Optimization finished")

    def _do_iteration(self, X, y, pbar):
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
            **parameters
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
                other_scores = dict(zip([n.name for n in self.default_objectives], other_scores))
            except Exception as e:
                pbar.write(str(e))
                score = np.nan
                other_scores = dict(zip([n.name for n in self.default_objectives], [np.nan] * len(self.default_objectives)))

            other_scores[self.objective.name] = score

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
        proposal = dict(zip([s[0] for s in space], values))
        return proposal

    def _add_result(self, trained_pipeline, parameters, scores, all_objective_scores, training_time):
        if self.objective.greater_is_better:
            score = min(scores)  # take worst across folds
            score_to_minimize = -score
        else:
            score = max(scores)  # take worst across folds
            score_to_minimize = score

        self.tuners[trained_pipeline.name].add(parameters.values(), score_to_minimize)

        # calculate high_variance_cv
        s = pd.Series(scores)
        s_mean, s_std = s.mean(), s.std()
        high, low = s_mean + 2 * s_std, s_mean - 2 * s_std
        high_variance_cv = (~s.between(left=low, right=high)).sum() > 0

        pipeline_name = trained_pipeline.__class__.__name__
        pipeline_id = len(self.results)

        self.results[pipeline_id] = {
            "id": pipeline_id,
            "pipeline_name": pipeline_name,
            "parameters": parameters,
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
            self._log("Warning! High variance between cross validation scores. " +
                      "Model may not perform as estimated on unseen data.")

        all_objective_scores = pd.DataFrame(pipeline_results["all_objective_scores"])
        mean = all_objective_scores.mean(axis=0)
        std = all_objective_scores.std(axis=0)
        all_objective_scores.loc["mean"] = mean
        all_objective_scores.loc["std"] = std

        with pd.option_context('display.float_format', '{:.3f}'.format):
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


if __name__ == "__main__":
    from evalml.objectives import FraudDetection
    from evalml.preprocessing import load_data

    filepath = "/Users/kanter/Documents/lead_scoring_app/fraud_demo/data/transactions.csv"
    X, y = load_data(filepath, index="id", label="fraud")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    from sklearn.datasets import load_digits

    digits = load_digits()

    X_train, X_test, y_train, y_test = split_data(pd.DataFrame(digits.data), pd.Series(digits.target), test_size=.2, random_state=0)
    print(X_train)
    objective = FraudDetection(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75,
        amount_col=10
    )

    clf = AutoClassifier(objective="precision",
                         max_pipelines=3,
                         random_state=0)

    clf.fit(X_train, y_train)

    print(clf.rankings)

    print(clf.best_pipeline)
    print(clf.best_pipeline.score(X_test, y_test))

    clf.rankings.to_csv("rankings.csv")
