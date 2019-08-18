# from evalml.pipelines import get_pipelines_by_model_type
import random
import time
from sys import stdout

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .auto_base import AutoBase

from evalml.objectives import get_objective
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
                 random_state=0,
                 tuner=None,
                 verbose=True):
        """Automated classifier pipeline search

        Arguments:
            objective (Object): the objective to optimize
            max_pipelines (int): maximum number of models to search
            max_time (int): maximum time in seconds to search for models
            model_types (list): The model types to search. By default searches over
                model_types
            cv (): cross validation method to use. By default StratifiedKFold
            tuner (): the tuner class to use. Defaults to scikit-optimize tuner
            random_state ():
            verbose (boolean): If True, turn verbosity on. Defaults to True

        """
        if objective is None:
            objective = "precision"

        if tuner is None:
            tuner = SKOptTuner

        self.objective = get_objective(objective)
        self.max_pipelines = max_pipelines
        self.max_time = max_time
        self.model_types = model_types
        self.verbose = verbose
        self.results = []
        self.trained_pipelines = {}
        self.random_state = random_state

        if cv is None:
            cv = StratifiedKFold(n_splits=3, random_state=random_state)
        self.cv = cv

        self.possible_pipelines = get_pipelines(model_types=model_types)

        self.tuners = {}
        self.search_spaces = {}
        for p in self.possible_pipelines:
            space = p.hyperparameters.copy()
            self.tuners[p.name] = tuner(space.values(), random_state=random_state)
            self.search_spaces[p.name] = space

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
            self._log("Searching up to %s pipelines" % self.max_pipelines)
            self._log("Optimizing for %s" % self.objective.name)
            self._log("")

        pbar = tqdm(range(self.max_pipelines), disable=not self.verbose, file=stdout)
        for n in pbar:
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

            # todo, how to provide metric. does it get optimize with the pipeline
            pbar.set_description("Testing %s" % (pipeline_class.name))
            # pbar.write("Parameters")
            # for item in parameters.items():
            #     pbar.write("%s: %s" % item)

            start = time.time()
            # todo improve CV
            scores = []
            for train, test in self.cv.split(X, y):
                if isinstance(X, pd.DataFrame):
                    X_train, X_test = X.iloc[train], X.iloc[test]
                else:
                    X_train, X_test = X[train], X[test]

                if isinstance(y, pd.Series):
                    y_train, y_test = y.iloc[train], y.iloc[test]
                else:
                    y_train, y_test = y[train], y[test]

                pipeline.fit(X_train, y_train)

                score = pipeline.score(X_test, y_test)
                scores.append(score)

            training_time = time.time() - start

            # save the result and continue
            self._add_result(
                trained_pipeline=pipeline,
                parameters=parameters,
                scores=scores,
                training_time=training_time
            )

        pbar.set_description("Optimization finished!")
        pbar.refresh()
        pbar.close()

    def _select_pipeline(self):
        return random.choice(self.possible_pipelines)

    def _propose_parameters(self, pipeline_class):
        values = self.tuners[pipeline_class.name].propose()
        space = self.search_spaces[pipeline_class.name]
        proposal = dict(zip(space.keys(), values))
        return proposal

    def _add_result(self, trained_pipeline, parameters, scores, training_time):
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
        high, low = s_mean + s_std, s_mean - s_std
        high_variance_cv = (~s.between(left=low, right=high)).sum() > 1

        pipeline_name = trained_pipeline.__class__.__name__
        to_add = {
            "id": len(self.results),
            "pipeline_name": pipeline_name,
            "parameters": parameters,
            "score": score,
            "scores": scores,
            "high_variance_cv": high_variance_cv,
            "training_time": training_time,
        }

        self.results.append(to_add)

        self._save_pipeline(pipeline_name, parameters, trained_pipeline)

    def _save_pipeline(self, pipeline_name, parameters, trained_pipeline):
        model_key = (pipeline_name, frozenset(parameters.items()))
        self.trained_pipelines[model_key] = trained_pipeline

    def get_pipeline(self, id):
        pipeline = None
        for r in self.results:
            if r["id"] == id:
                pipeline = r

        if pipeline is None:
            raise RuntimeError("Pipeline not found")

        model_key = (pipeline['pipeline_name'], frozenset(pipeline['parameters'].items()))
        return self.trained_pipelines[model_key]

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""
        ascending = True
        if self.objective.greater_is_better:
            ascending = False

        rankings_df = pd.DataFrame(self.results)
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
