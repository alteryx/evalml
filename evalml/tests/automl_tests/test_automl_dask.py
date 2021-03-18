import unittest

import numpy as np
import pytest
from distributed import Client

from evalml.automl import AutoMLSearch
from evalml.automl.engine import DaskEngine, SequentialEngine


@pytest.mark.usefixtures("X_y_binary_cls")
class TestAutoMLSearchDask(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = Client()
        cls.parallel_engine = DaskEngine(cls.client)
        cls.sequential_engine = SequentialEngine()

    def test_automl(self):
        """ Comparing the results of parallel and sequential AutoML to each other."""
        X, y = self.X_y_binary
        par_automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.parallel_engine)
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.sequential_engine)
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        parallel_results = parallel_rankings.drop(columns=["id"])
        sequential_results = sequential_rankings.drop(columns=["id"])

        assert parallel_results.drop(columns=["validation_score"]).equals(
            sequential_results.drop(columns=["validation_score"]))
        assert np.allclose(np.array(sequential_results["validation_score"]),
                           np.array(parallel_results["validation_score"]))

    def test_automl_max_iterations(self):
        """ Making sure that the max_iterations parameter limits the number of pipelines run. """
        X, y = self.X_y_binary
        max_iterations = 4
        par_automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.parallel_engine,
                                  max_iterations=max_iterations)
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.sequential_engine,
                                  max_iterations=max_iterations)
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        assert len(sequential_rankings) == len(parallel_rankings) == max_iterations
        # TODO: Figure out how to mock the train_and_score_pipelines call to assert the call count.

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
