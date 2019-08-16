# from evalml.pipelines import get_pipelines_by_model_type
import time

import pandas as pd
from tqdm import tqdm

import evalml.pipelines


class AutoClassifier:
    def __init__(self, max_models=5, max_time=60 * 5, model_types=None, verbose=True):
        """Automated classifer search

        Arguments:
            max_models (int): maximum number of models to search
            max_time (int): maximum time in seconds to search for models
            model_types (list): The model types to search. By default searches over
                model_types

        """
        self.max_models = max_models
        self.max_time = max_time
        self.model_types = model_types
        self.verbose = verbose
        self.results = []
        self.trained_pipelines = {}

    def fit(self, X, y, metric=None, feature_types=None):
        """Find best classifier

        Arguments:
            X (pd.DataFrame): the input training data of shape [n_samples, n_features]

            y (pd.Series): the target training labels of length [n_samples]

            metric (Metric): the metric to optimize

            feature_types (list, optional): list of feature types. either numeric of categorical.
                categorical features will automatically be encoded

        Returns:

            self
        """

        pbar = tqdm(range(self.max_models), disable=not self.verbose)
        for n in pbar:
            # determine which pipeline to build
            pipeline_class = self._select_pipeline()

            # propose the next best parameters for this piepline
            parameters = self._propose_parameters(pipeline_class)

            # fit an score the piepline
            pipeline = pipeline_class(**parameters)

            # todo, how to provide metric. does it get optimize with the pipeline
            start = time.time()
            pipeline.fit(X, y)
            score = pipeline.score(X, y)
            training_time = time.time() - start

            pbar.write("Best so far: %f" % score)

            # save the result and continue
            self._add_result(
                trained_pipeline=pipeline,
                parameters=parameters,
                score=score,
                training_time=training_time
            )

    def _select_pipeline(self):
        # possible_pipelines = get_pipelines_by_model_type(self.model_types)
        return evalml.pipelines.RFPipeline

    def _propose_parameters(self, pipeline_class):
        parameters = pipeline_class.get_hyperparameters()
        _ = parameters

        # proposal = {}
        # for p in parameters:
        #     proposal[p] = 10 # make this actually sample

        proposal = {
            "n_estimators": 10,
            "strategy": "mean"
        }

        return proposal

    def _add_result(self, trained_pipeline, parameters, score, training_time):
        pipeline_name = trained_pipeline.__class__.__name__
        to_add = {
            "pipeline_name": pipeline_name,
            "parameters": parameters,
            "score": score,
            "training_time": training_time,
        }

        self.results.append(to_add)

        self._save_pipeline(pipeline_name, parameters, trained_pipeline)

    def _save_pipeline(self, pipeline_name, parameters, trained_pipeline):
        model_key = (pipeline_name, frozenset(parameters.items()))
        self.trained_pipelines[model_key] = trained_pipeline

    def _get_pipeline(self, pipeline_name, parameters):
        model_key = (pipeline_name, frozenset(parameters.items()))
        return self.trained_pipelines[model_key]

    @property
    def rankings(self):
        """Returns the rankings of the models searched"""

        rankings_df = pd.DataFrame(self.results).sort_values("score")
        return rankings_df

    @property
    def best_model(self):
        """Returns the best model found"""
        best = self.rankings.iloc[0]
        return self._get_pipeline(best["pipeline_name"], best["parameters"])


if __name__ == "__main__":
    from evalml.objectives import FraudDetection
    from evalml.preprocessing import load_data, split_data

    filepath = "/Users/kanter/Documents/lead_scoring_app/fraud_demo/data/transactions.csv"
    X, y = load_data(filepath, index="id", label="fraud")

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    X = X.select_dtypes(include=numerics)

    x_train, x_test, y_train, y_test = split_data(X, y, test_size=.2, random_state=0)

    objective = FraudDetection(
        retry_percentage=.5,
        interchange_fee=.02,
        fraud_payout_percentage=.75
    )

    clf = AutoClassifier(max_models=3)

    clf.fit(X, y, objective)

    print(clf.rankings)

    print(clf.best_model)
