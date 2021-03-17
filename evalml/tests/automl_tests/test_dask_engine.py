import pytest
import time
# import dask.distributed as distributed
from distributed.utils_test import client, loop, cluster_fixture

import woodwork as ww
from evalml.automl.engine.dask_engine import DaskEngine, DaskComputation
from evalml.automl.engine.sequential_engine import SequentialEngine
from evalml.tests.automl_tests.dask_testing import automl_data, TestLRCPipeline, TestSVMPipeline, TestCBPipeline


def test_init(client):
    engine = DaskEngine(client=client)
    assert engine.client == client

    with pytest.raises(TypeError, match="Expected dask.distributed.Client, received"):
        DaskEngine(client="Client")


# TODO: Might need to add this:
"""
evalml/tests/automl_tests/test_dask_engine.py::test_submit_training_job_single
  Coroutine functions are not natively supported and have been skipped.
  You need to install a suitable plugin for your async framework, for example:
    - pytest-asyncio
    - pytest-trio
    - pytest-tornasync
"""


def test_submit_training_job_single(client, X_y_binary):
    X, y = X_y_binary
    pipeline = TestLRCPipeline({})

    engine = DaskEngine(client=client)
    pipeline_future = engine.submit_training_job(X=X, y=y, automl_data=automl_data, pipeline=pipeline)
    pipeline_fitted = pipeline_future.get_result()
    assert pipeline_fitted._is_fitted


def test_submit_training_jobs_multiple(client, X_y_binary):
    X, y = X_y_binary
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
    par_pipelines, par_time_taken = fit_pipelines(pipelines, DaskEngine(client=client))
    for pipeline in par_pipelines:
        assert pipeline._is_fitted

    # Ensure sequential and parallel pipelines are equivalent
    assert len(par_pipelines) == len(seq_pipelines)
    for par_pipeline in par_pipelines:
        assert par_pipeline in seq_pipelines

    print(seq_time_taken, par_time_taken)


def test_submit_evaluate_job_single(client, X_y_binary):
    X, y = X_y_binary
    pipeline = TestLRCPipeline({})
    engine = DaskEngine(client=client)

    pipeline_future = engine.submit_evaluation_job(X=ww.DataTable(X), y=ww.DataColumn(y),
                                                   automl_data=automl_data, pipeline=pipeline)
    assert isinstance(pipeline_future, DaskComputation)
    pipeline_results = pipeline_future.get_result()


def test_submit_training_jobs_multiple(client, X_y_binary):
    X, y = X_y_binary
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

    # Check that pipelines are fitted
    par_eval_results, par_time_taken = eval_pipelines(pipelines, DaskEngine(client=client))
    par_dicts = [s[0] for s in par_eval_results]
    par_pipelines = [s[1] for s in par_eval_results]
    par_loggers = [s[2] for s in par_eval_results]

    assert all([s._is_fitted for s in par_pipelines])
    assert len(par_eval_results) == len(
        pipelines)  # Check that number of parallel evaluated pipelines matches original number of pipelines.
