import unittest

import numpy as np
import pytest
from distributed import Client

from evalml.automl import AutoMLSearch
from evalml.automl.callbacks import raise_error_callback
from evalml.automl.engine import DaskEngine, SequentialEngine
from evalml.tests.automl_tests.dask_test_utils import (
    TestPipelineFast,
    TestPipelineSlow,
    TestPipelineWithFitError,
    TestPipelineWithScoreError
)


@pytest.mark.usefixtures("X_y_binary_cls")
class TestAutoMLSearchDask(unittest.TestCase):

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        """ Gives the unittests access to the logger"""
        self._caplog = caplog

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

        par_results = parallel_rankings.drop(columns=["id"])
        seq_results = sequential_rankings.drop(columns=["id"])

        assert all(seq_results["pipeline_name"] == par_results["pipeline_name"])
        assert np.allclose(np.array(seq_results["mean_cv_score"]), np.array(par_results["mean_cv_score"]))
        assert np.allclose(np.array(seq_results["validation_score"]), np.array(par_results["validation_score"]))
        assert np.allclose(np.array(seq_results["percent_better_than_baseline"]), np.array(par_results["percent_better_than_baseline"]))

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

    def test_automl_train_dask_error_callback(self):
        """ Make sure the pipeline training error message makes its way back from the workers. """
        self._caplog.clear()
        X, y = self.X_y_binary
        pipelines = [TestPipelineWithFitError({})]
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.parallel_engine,
                              max_iterations=2, allowed_pipelines=pipelines)
        automl.train_pipelines(pipelines)
        assert "Train error for PipelineWithError: Yikes" in self._caplog.text

    def test_automl_score_dask_error_callback(self):
        """ Make sure the pipeline scoring error message makes its way back from the workers. """
        self._caplog.clear()
        X, y = self.X_y_binary
        pipelines = [TestPipelineWithScoreError({})]
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.parallel_engine,
                              max_iterations=2, allowed_pipelines=pipelines)
        automl.score_pipelines(pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"])
        assert "Score error for PipelineWithError" in self._caplog.text

    def test_automl_immediate_quit(self):
        """ Make sure the AutoMLSearch quits when error_callback is defined and does no further work. """
        self._caplog.clear()
        X, y = self.X_y_binary
        pipelines = [TestPipelineFast({}), TestPipelineWithFitError({}), TestPipelineSlow({})]
        automl = AutoMLSearch(X_train=X, y_train=y, problem_type="binary", engine=self.parallel_engine,
                              max_iterations=4, allowed_pipelines=pipelines, error_callback=raise_error_callback,
                              optimize_thresholds=False)

        # Ensure the broken pipeline raises the error
        with pytest.raises(Exception, match="Yikes"):
            automl.search()

        # Make sure the automl algorithm stopped after the broken pipeline raised
        assert len(automl.full_rankings) < len(pipelines)
        assert TestPipelineFast.custom_name in set(automl.full_rankings["pipeline_name"])
        assert TestPipelineSlow.custom_name not in set(automl.full_rankings["pipeline_name"])
        assert TestPipelineWithFitError.custom_name not in set(automl.full_rankings["pipeline_name"])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
