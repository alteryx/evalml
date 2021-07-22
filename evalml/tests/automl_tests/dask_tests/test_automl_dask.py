from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pytest
from dask import distributed as dd
from dask.distributed import LocalCluster

from evalml.automl import AutoMLSearch
from evalml.automl.automl_algorithm import IterativeAlgorithm
from evalml.automl.callbacks import raise_error_callback
from evalml.automl.engine import CFEngine, DaskEngine, SequentialEngine
from evalml.automl.engine.cf_engine import CFClient
from evalml.tests.automl_tests.dask_test_utils import (
    DaskPipelineFast,
    DaskPipelineSlow,
    DaskPipelineWithFitError,
    DaskPipelineWithScoreError,
)
from evalml.tuners import SKOptTuner


@pytest.fixture
def sequential_engine():
    return SequentialEngine()


@pytest.fixture(scope="module")
def cluster():
    dask_cluster = LocalCluster(n_workers=1, dashboard_address=None)
    yield dask_cluster
    dask_cluster.close()


@pytest.fixture(scope="module")
def thread_pool():
    pool = ThreadPoolExecutor()
    yield pool
    pool.shutdown()


@pytest.fixture(scope="module")
def process_pool():
    pool = ProcessPoolExecutor()
    yield pool
    pool.shutdown()


# List of tuples indicating which engine types and resource pools to parametrize tests over.
# Removed '("CFEngine", "process")' as it doesn't work properly on GitHub test runners.
engine_and_resource_types = [("CFEngine", "thread"), ("DaskEngine", "N/A")]


def _get_engine_support(parallel_engine_type, thread_pool, cluster):
    """Helper function to return the proper combination of resource pool, client class and
    engine class for testing purposes.

    e.g. The CFEngine can be run either with a ThreadPoolExecutor or a ProcessPoolExecutor,
        so _get_engine_support("CFEngine", thread_pool, cluster) returns a
        tuple of (ThreadPoolExecutor, cf.Client, cf.CFEngine)
    """
    if parallel_engine_type == "CFEngine":
        resources = thread_pool
        client_class = CFClient
        engine_class = CFEngine
    elif parallel_engine_type == "DaskEngine":
        resources = cluster
        client_class = dd.Client
        engine_class = DaskEngine
    return resources, client_class, engine_class


@pytest.mark.parametrize(
    "parallel_engine_type,pool_type",
    engine_and_resource_types,
)
def test_automl(
    parallel_engine_type,
    pool_type,
    X_y_binary_cls,
    cluster,
    thread_pool,
    process_pool,
    sequential_engine,
):
    """Comparing the results of parallel and sequential AutoML to each other."""

    resources, client_class, engine_class = _get_engine_support(
        parallel_engine_type, thread_pool, cluster
    )

    with client_class(resources) as client:
        parallel_engine = engine_class(client)
        X, y = X_y_binary_cls
        par_automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=parallel_engine
        )
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(
            X_train=X, y_train=y, problem_type="binary", engine=sequential_engine
        )
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        par_results = parallel_rankings.drop(columns=["id"])
        seq_results = sequential_rankings.drop(columns=["id"])

        assert all(seq_results["pipeline_name"] == par_results["pipeline_name"])
        assert np.allclose(
            np.array(seq_results["mean_cv_score"]),
            np.array(par_results["mean_cv_score"]),
        )
        assert np.allclose(
            np.array(seq_results["validation_score"]),
            np.array(par_results["validation_score"]),
        )
        assert np.allclose(
            np.array(seq_results["percent_better_than_baseline"]),
            np.array(par_results["percent_better_than_baseline"]),
        )


@pytest.mark.parametrize(
    "parallel_engine_type,pool_type",
    engine_and_resource_types,
)
def test_automl_max_iterations(
    parallel_engine_type,
    pool_type,
    X_y_binary_cls,
    cluster,
    thread_pool,
    process_pool,
    sequential_engine,
):
    """Making sure that the max_iterations parameter limits the number of pipelines run."""

    X, y = X_y_binary_cls
    resources, client_class, engine_class = _get_engine_support(
        parallel_engine_type, thread_pool, cluster
    )

    with client_class(resources) as client:
        parallel_engine = engine_class(client)

        max_iterations = 4
        par_automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=max_iterations,
        )
        par_automl.search()
        parallel_rankings = par_automl.full_rankings

        seq_automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=sequential_engine,
            max_iterations=max_iterations,
        )
        seq_automl.search()
        sequential_rankings = seq_automl.full_rankings

        assert len(sequential_rankings) == len(parallel_rankings) == max_iterations


@pytest.mark.parametrize(
    "parallel_engine_type,pool_type",
    engine_and_resource_types,
)
def test_automl_train_dask_error_callback(
    parallel_engine_type,
    pool_type,
    X_y_binary_cls,
    cluster,
    thread_pool,
    process_pool,
    sequential_engine,
    caplog,
):
    """Make sure the pipeline training error message makes its way back from the workers."""
    caplog.clear()
    X, y = X_y_binary_cls
    resources, client_class, engine_class = _get_engine_support(
        parallel_engine_type, thread_pool, cluster
    )

    with client_class(resources) as client:
        parallel_engine = engine_class(client)

        pipelines = [DaskPipelineWithFitError({})]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=2,
        )
        automl.allowed_pipelines = pipelines

        automl.train_pipelines(pipelines)
        assert "Train error for PipelineWithError: Yikes" in caplog.text


@pytest.mark.parametrize(
    "parallel_engine_type,pool_type",
    engine_and_resource_types,
)
def test_automl_score_dask_error_callback(
    parallel_engine_type,
    pool_type,
    X_y_binary_cls,
    cluster,
    thread_pool,
    process_pool,
    sequential_engine,
    caplog,
):
    """Make sure the pipeline scoring error message makes its way back from the workers."""
    caplog.clear()
    X, y = X_y_binary_cls
    resources, client_class, engine_class = _get_engine_support(
        parallel_engine_type, thread_pool, cluster
    )

    with client_class(resources) as client:
        parallel_engine = engine_class(client)
        pipelines = [DaskPipelineWithScoreError({})]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=2,
        )
        automl.allowed_pipelines = pipelines

        automl.score_pipelines(
            pipelines, X, y, objectives=["Log Loss Binary", "F1", "AUC"]
        )
        assert "Score error for PipelineWithError" in caplog.text


@pytest.mark.parametrize(
    "parallel_engine_type,pool_type",
    engine_and_resource_types,
)
def test_automl_immediate_quit(
    parallel_engine_type,
    pool_type,
    X_y_binary_cls,
    cluster,
    thread_pool,
    process_pool,
    sequential_engine,
    caplog,
):
    """Make sure the AutoMLSearch quits when error_callback is defined and does no further work."""
    caplog.clear()
    X, y = X_y_binary_cls
    resources, client_class, engine_class = _get_engine_support(
        parallel_engine_type, thread_pool, cluster
    )

    with client_class(resources) as client:
        parallel_engine = engine_class(client)

        pipelines = [
            DaskPipelineFast({}),
            DaskPipelineWithFitError({}),
            DaskPipelineSlow({}),
        ]
        automl = AutoMLSearch(
            X_train=X,
            y_train=y,
            problem_type="binary",
            engine=parallel_engine,
            max_iterations=4,
            error_callback=raise_error_callback,
            optimize_thresholds=False,
        )
        automl._automl_algorithm = IterativeAlgorithm(
            max_iterations=4,
            allowed_pipelines=pipelines,
            tuner_class=SKOptTuner,
            random_seed=0,
            n_jobs=-1,
            number_features=X.shape[1],
            pipelines_per_batch=5,
            ensembling=False,
            text_in_ensembling=False,
            pipeline_params={},
            custom_hyperparameters=None,
        )

        # Ensure the broken pipeline raises the error
        with pytest.raises(Exception, match="Yikes"):
            automl.search()

        # Make sure the automl algorithm stopped after the broken pipeline raised
        assert len(automl.full_rankings) < len(pipelines)
        assert DaskPipelineSlow.custom_name not in set(
            automl.full_rankings["pipeline_name"]
        )
        assert DaskPipelineWithFitError.custom_name not in set(
            automl.full_rankings["pipeline_name"]
        )
