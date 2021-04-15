import unittest

import numpy as np
import pytest
import woodwork as ww
from distributed import Client

from evalml.automl.engine.dask_engine import DaskComputation, DaskEngine
from evalml.automl.engine.engine_base import (
    JobLogger,
    evaluate_pipeline,
    train_pipeline
)
from evalml.automl.engine.sequential_engine import SequentialEngine
from evalml.pipelines import BinaryClassificationPipeline
from evalml.pipelines.pipeline_base import PipelineBase
from evalml.tests.automl_tests.dask_test_utils import (
    TestPipelineSlow,
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
        """ Test that training a single pipeline using the parallel engine produces the
        same results as simply running the train_pipeline function. """
        X, y = self.X_y_binary
        engine = DaskEngine(client=self.client)
        pipeline = BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

        # Verify that engine fits a pipeline
        pipeline_future = engine.submit_training_job(X=X, y=y, automl_config=automl_data, pipeline=pipeline)
        dask_pipeline_fitted = pipeline_future.get_result()
        assert dask_pipeline_fitted._is_fitted

        # Verify parallelization has no effect on output of function
        original_pipeline_fitted = train_pipeline(pipeline, X, y, optimize_thresholds=automl_data.optimize_thresholds,
                                                  objective=automl_data.objective)
        assert dask_pipeline_fitted == original_pipeline_fitted
        assert dask_pipeline_fitted.predict(X) == original_pipeline_fitted.predict(X)

    def test_submit_training_jobs_multiple(self):
        """ Test that training multiple pipelines using the parallel engine produces the
        same results as the sequential engine. """
        X, y = self.X_y_binary
        pipelines = [BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                  parameters={"Logistic Regression Classifier": {"n_jobs": 1}}),
                     BinaryClassificationPipeline(component_graph=["Baseline Classifier"]),
                     BinaryClassificationPipeline(component_graph=["SVM Classifier"])]

        def fit_pipelines(pipelines, engine):
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_training_job(X=X, y=y, automl_config=automl_data, pipeline=pipeline))
            results = [f.get_result() for f in futures]
            return results

        # Verify all pipelines are trained and fitted.
        seq_pipelines = fit_pipelines(pipelines, SequentialEngine())
        for pipeline in seq_pipelines:
            assert pipeline._is_fitted

        # Verify all pipelines are trained and fitted.
        par_pipelines = fit_pipelines(pipelines, DaskEngine(client=self.client))
        for pipeline in par_pipelines:
            assert pipeline._is_fitted

        # Ensure sequential and parallel pipelines are equivalent
        assert len(par_pipelines) == len(seq_pipelines)
        for par_pipeline in par_pipelines:
            assert par_pipeline in seq_pipelines

    def test_submit_evaluate_job_single(self):
        """ Test that evaluating a single pipeline using the parallel engine produces the
        same results as simply running the evaluate_pipeline function. """
        X, y = self.X_y_binary
        X = ww.DataTable(X)
        y = ww.DataColumn(y)
        pipeline = BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                parameters={"Logistic Regression Classifier": {"n_jobs": 1}})

        engine = DaskEngine(client=self.client)

        # Verify that engine evaluates a pipeline
        pipeline_future = engine.submit_evaluation_job(X=X, y=y,
                                                       automl_config=automl_data, pipeline=pipeline)
        assert isinstance(pipeline_future, DaskComputation)

        par_eval_results = pipeline_future.get_result()

        original_eval_results = evaluate_pipeline(pipeline, automl_config=automl_data, X=X, y=y, logger=JobLogger())

        # Ensure we get back the same output as the parallelized function.
        assert len(par_eval_results) == 3

        par_scores = par_eval_results.get("scores")
        original_eval_scores = original_eval_results.get("scores")

        # Compare cross validation information except training time.
        assert par_scores["cv_data"] == original_eval_scores["cv_data"]
        assert all(par_scores["cv_scores"] == original_eval_scores["cv_scores"])
        assert par_scores["cv_score_mean"] == par_scores["cv_score_mean"]

        # Make sure the resulting pipelines are the same.
        assert isinstance(par_eval_results.get("pipeline"), PipelineBase)
        assert par_eval_results.get("pipeline") == original_eval_results.get("pipeline")

        # Make sure a properly filled logger comes back.
        assert isinstance(par_eval_results.get("logger"), JobLogger)
        assert par_eval_results.get("logger").logs == original_eval_results.get("logger").logs

    def test_submit_evaluate_jobs_multiple(self):
        """ Test that evaluating multiple pipelines using the parallel engine produces the
        same results as the sequential engine. """
        X, y = self.X_y_binary
        pipelines = [BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                  parameters={"Logistic Regression Classifier": {"n_jobs": 1}}),
                     BinaryClassificationPipeline(component_graph=["Baseline Classifier"]),
                     BinaryClassificationPipeline(component_graph=["SVM Classifier"])]

        def eval_pipelines(pipelines, engine):
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_evaluation_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                            automl_config=automl_data, pipeline=pipeline))
            results = [f.get_result() for f in futures]
            return results

        par_eval_results = eval_pipelines(pipelines, DaskEngine(client=self.client))
        par_dicts = [s.get("scores") for s in par_eval_results]
        par_scores = [s["cv_data"][0]["mean_cv_score"] for s in par_dicts]
        par_pipelines = [s.get("pipeline") for s in par_eval_results]

        seq_eval_results = eval_pipelines(pipelines, SequentialEngine())
        seq_dicts = [s.get("scores") for s in seq_eval_results]
        seq_scores = [s["cv_data"][0]["mean_cv_score"] for s in seq_dicts]
        seq_pipelines = [s.get("pipeline") for s in seq_eval_results]

        # Ensure all pipelines are fitted.
        assert all([s._is_fitted for s in par_pipelines])

        # Ensure the scores in parallel and sequence are same
        assert set(par_scores) == set(seq_scores)
        assert not any([np.isnan(s) for s in par_scores])

        # Ensure the parallel and sequence pipelines match
        assert len(par_pipelines) == len(seq_pipelines)
        for par_pipeline in par_pipelines:
            assert par_pipeline in seq_pipelines

    def test_submit_scoring_job_single(self):
        """ Test that scoring a single pipeline using the parallel engine produces the
        same results as simply running the score_pipeline function. """
        X, y = self.X_y_binary
        pipeline = BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                parameters={"Logistic Regression Classifier": {"n_jobs": 1}})
        engine = DaskEngine(client=self.client)
        objectives = [automl_data.objective]

        pipeline_future = engine.submit_training_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                     automl_config=automl_data, pipeline=pipeline)
        pipeline = pipeline_future.get_result()
        pipeline_score_future = engine.submit_scoring_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                          automl_config=automl_data, pipeline=pipeline,
                                                          objectives=objectives)
        assert isinstance(pipeline_score_future, DaskComputation)
        pipeline_score = pipeline_score_future.get_result()

        original_pipeline_score = pipeline.score(X=X, y=y, objectives=objectives)

        assert not np.isnan(pipeline_score["Log Loss Binary"])
        assert pipeline_score == original_pipeline_score

    def test_submit_scoring_jobs_multiple(self):
        """ Test that scoring multiple pipelines using the parallel engine produces the
        same results as the sequential engine. """
        X, y = self.X_y_binary
        pipelines = [BinaryClassificationPipeline(component_graph=["Logistic Regression Classifier"],
                                                  parameters={"Logistic Regression Classifier": {"n_jobs": 1}}),
                     BinaryClassificationPipeline(component_graph=["Baseline Classifier"]),
                     BinaryClassificationPipeline(component_graph=["SVM Classifier"])]

        def score_pipelines(pipelines, engine):
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_training_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                          automl_config=automl_data, pipeline=pipeline))
            pipelines = [f.get_result() for f in futures]
            futures = []
            for pipeline in pipelines:
                futures.append(engine.submit_scoring_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                         automl_config=automl_data, pipeline=pipeline,
                                                         objectives=[automl_data.objective]))
            results = [f.get_result() for f in futures]
            return results

        par_eval_results = score_pipelines(pipelines, DaskEngine(client=self.client))
        par_scores = [s["Log Loss Binary"] for s in par_eval_results]

        seq_eval_results = score_pipelines(pipelines, SequentialEngine())
        seq_scores = [s["Log Loss Binary"] for s in seq_eval_results]

        # Check there are the proper number of pipelines and all their scores are same.
        assert len(par_eval_results) == len(pipelines)
        assert set(par_scores) == set(seq_scores)

    def test_cancel_job(self):
        """ Test that training a single pipeline using the parallel engine produces the
                same results as simply running the train_pipeline function. """
        X, y = self.X_y_binary
        engine = DaskEngine(client=self.client)
        pipeline = TestPipelineSlow({"Logistic Regression Classifier": {"n_jobs": 1}})

        # Verify that engine fits a pipeline
        pipeline_future = engine.submit_training_job(X=X, y=y, automl_config=automl_data, pipeline=pipeline)
        pipeline_future.cancel()
        assert pipeline_future.is_cancelled

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()
