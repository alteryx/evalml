import time
import unittest
import pytest
import numpy as np
from distributed import Client

import woodwork as ww

from evalml.automl.engine.dask_engine import DaskComputation, DaskEngine
from evalml.automl.engine.sequential_engine import SequentialEngine
from evalml.tests.automl_tests.dask_testing import (
    TestCBPipeline,
    TestLRCPipeline,
    TestSVMPipeline,
    automl_data
)


@pytest.mark.usefixtures("X_y_binary_cls")
class TestDaskEngine(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = Client()

    def test_init(self):
        engine = DaskEngine(client=self.client)
        assert engine.client == self.client

        with pytest.raises(TypeError, match="Expected dask.distributed.Client, received"):
            DaskEngine(client="Client")

    def test_submit_training_job_single(self):
        X, y = self.X_y_binary
        pipeline = TestLRCPipeline({})

        engine = DaskEngine(client=self.client)
        pipeline_future = engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline)
        pipeline_fitted = pipeline_future.get_result()
        assert pipeline_fitted._is_fitted

    def test_submit_training_jobs_multiple(self):
        X, y = self.X_y_binary
        pipelines = [TestLRCPipeline({}),
                     TestCBPipeline({}),
                     TestSVMPipeline({})]

        def fit_pipelines(pipelines, engine):
            time_start = time.time()
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline))
            results = [f.get_result() for f in futures]
            time_taken = time.time() - time_start
            return results, time_taken

        seq_pipelines, seq_time_taken = fit_pipelines(pipelines, SequentialEngine())
        for pipeline in seq_pipelines:
            assert pipeline._is_fitted

        # Check that pipelines are fitted
        par_pipelines, par_time_taken = fit_pipelines(pipelines, DaskEngine(client=self.client))
        for pipeline in par_pipelines:
            assert pipeline._is_fitted

        # Ensure sequential and parallel pipelines are equivalent
        assert len(par_pipelines) == len(seq_pipelines)
        for par_pipeline in par_pipelines:
            assert par_pipeline in seq_pipelines

        print(seq_time_taken, par_time_taken)

    def test_submit_evaluate_job_single(self):
        X, y = self.X_y_binary
        pipeline = TestLRCPipeline({})
        engine = DaskEngine(client=self.client)

        pipeline_future = engine.submit_evaluation_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                       automl_data=automl_data, pipeline=pipeline)
        assert isinstance(pipeline_future, DaskComputation)
        pipeline_results = pipeline_future.get_result()

    def test_submit_evaluate_jobs_multiple(self):
        X, y = self.X_y_binary
        pipelines = [TestLRCPipeline({}),
                     TestCBPipeline({}),
                     TestSVMPipeline({})]

        def eval_pipelines(pipelines, engine):
            time_start = time.time()
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_evaluation_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                            automl_data=automl_data, pipeline=pipeline))
            results = [f.get_result() for f in futures]
            time_taken = time.time() - time_start
            return results, time_taken

        par_eval_results, par_time_taken = eval_pipelines(pipelines, DaskEngine(client=self.client))

        # Extract result information from parallel evaluation
        par_dicts = [s[0] for s in par_eval_results]
        par_scores = [s["cv_data"][0]["score"] for s in par_dicts]
        par_pipelines = [s[1] for s in par_eval_results]

        # Check all pipelines are fitted, there are the proper number of pipelines and all their scores are reasonable.
        assert all([s._is_fitted for s in par_pipelines])
        assert len(par_eval_results) == len(pipelines)
        assert not any([np.isnan(s) for s in par_scores])

    def test_submit_scoring_job_single(self):
        X, y = self.X_y_binary
        pipeline = TestLRCPipeline({})
        engine = DaskEngine(client=self.client)

        pipeline_future = engine.submit_training_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                     automl_data=automl_data, pipeline=pipeline)
        assert isinstance(pipeline_future, DaskComputation)
        pipeline = pipeline_future.get_result()
        pipeline_future = engine.submit_scoring_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                    automl_data=automl_data, pipeline=pipeline,
                                                    objectives=[automl_data.objective])
        assert isinstance(pipeline_future, DaskComputation)
        pipeline_results = pipeline_future.get_result()

        assert not np.isnan(pipeline_results["Log Loss Binary"])

    def test_submit_scoring_jobs_multiple(self):
        X, y = self.X_y_binary
        pipelines = [TestLRCPipeline({}),
                     TestCBPipeline({}),
                     TestSVMPipeline({})]

        def score_pipelines(pipelines, engine):
            time_start = time.time()
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_training_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                          automl_data=automl_data, pipeline=pipeline))
            pipelines = [f.get_result() for f in futures]
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_scoring_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                         automl_data=automl_data, pipeline=pipeline,
                                                         objectives=[automl_data.objective]))
            results = [f.get_result() for f in futures]
            time_taken = time.time() - time_start
            return results, time_taken

        par_eval_results, par_time_taken = score_pipelines(pipelines, DaskEngine(client=self.client))
        par_scores = [s["Log Loss Binary"] for s in par_eval_results]

        seq_eval_results, seq_time_taken = score_pipelines(pipelines, SequentialEngine())
        seq_scores = [s["Log Loss Binary"] for s in seq_eval_results]

        # Check there are the proper number of pipelines and all their scores are same.
        assert len(par_eval_results) == len(pipelines)
        assert set(par_scores) == set(seq_scores)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()


if __name__ == "__main__":
    unittest.main()
